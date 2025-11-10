import os
import json
import base64
from pathlib import Path
import time
from PIL import Image, ImageDraw
import requests
import torch
import cv2
import supervision as sv
from dotenv import load_dotenv
import os

from transformers import AutoImageProcessor, DetrForObjectDetection
from ultralytics import YOLO
# from mistralai import Mistral
# from mistralai.extra import response_format_from_pydantic_model
from pydantic import BaseModel, Field

try:
    import fitz  # PyMuPDF for PDF handling
except ImportError:
    raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")


# Pydantic model for OCR extraction
# class Document(BaseModel):
#     Name: str = Field(..., description="Name of the person mentioned in the document")
#     DoB: str = Field(..., description="Date of birth of the person")
#     Gender: str = Field(..., description="Applicant's gender")
#     Marital_Status: str = Field(..., description="Applicant's marital status")
#     Address: str = Field(..., description="Address of the person")
#     Aadhar_Number: str = Field(..., description="Aadhar number")
#     PAN_Number: str = Field(..., description="PAN number")
#     Bank_Account: str = Field(..., description="Applicant's bank account number")
#     GSTIN_Number: str = Field(..., description="GSTIN number mentioned in the document")
#     type_of_document: str = Field(..., description="Type of KYC document")


class ObjectDetection:
    def __init__(self):
        # Initialize stamp detection model
        print("Loading stamp detection model...")
        self.stamp_processor = AutoImageProcessor.from_pretrained("erikaxenia/detr-finetuned-stamp-v2")
        self.stamp_model = DetrForObjectDetection.from_pretrained("erikaxenia/detr-finetuned-stamp-v2")
        
        # Initialize signature detection YOLO model
        print("Loading signature detection model...")
        self.signature_model = YOLO("yolov8s.pt")
        
        # Initialize photo detection YOLO model
        print("Loading photo detection model...")
        self.photo_model = YOLO("yolov8s.pt")  # Same model or use different weights
        
        # Initialize OCR client
        print("Loading OCR client...")
        # load_dotenv()
        # mistral_api_key = os.getenv('MISTRAL_API_KEY')
        # self.ocr_client = Mistral(api_key=mistral_api_key) 
        print("✓ All models loaded successfully\n")

    def pdf_to_images(self, pdf_path):
        """Convert PDF pages to PIL Image objects"""
        pdf_path = Path(pdf_path).resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        doc = fitz.open(str(pdf_path))
        images = []
        for page in doc:
            pix = page.get_pixmap()
            mode = "RGB" if pix.n < 4 else "RGBA"
            img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
            if mode == "RGBA":
                img = img.convert("RGB")
            images.append(img)
        return images

    # ============ STAMP DETECTION ============
    def detect_stamps(self, pdf_path, output_dir, score_thresh=0.5):
        """Detect stamps in PDF using DETR model"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        images = self.pdf_to_images(pdf_path)
        results = {"pdf": str(pdf_path), "pages": []}

        for idx, img in enumerate(images, start=1):
            if img.mode != "RGB":
                img = img.convert("RGB")

            inputs = self.stamp_processor(images=img, return_tensors="pt")
            with torch.no_grad():
                outputs = self.stamp_model(**inputs)

            target_size = [(img.height, img.width)]
            detections = self.stamp_processor.post_process_object_detection(
                outputs, target_sizes=target_size, threshold=score_thresh)[0]

            img_with_boxes = img.copy()
            draw = ImageDraw.Draw(img_with_boxes)

            page_info = {"page": idx, "detections": []}
            for det_i, (box, score, label) in enumerate(zip(
                    detections["boxes"], detections["scores"], detections["labels"]), start=1):
                x0, y0, x1, y1 = [int(max(0, round(v))) for v in box.tolist()]
                if score > score_thresh:
                    crop = img.crop((x0, y0, x1, y1))
                    crop_name = f"{Path(pdf_path).stem}_page{idx}_stamp{det_i}.png"
                    crop_path = output_dir / crop_name
                    crop.save(crop_path)

                    draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
                    label_text = f"Stamp: {score:.2f}"
                    draw.text((x0, y0), label_text, fill="red")

                    page_info["detections"].append({
                        "id": det_i,
                        "label": int(label.item()) if hasattr(label, "item") else int(label),
                        "score": float(score.item()) if hasattr(score, "item") else float(score),
                        "box": [x0, y0, x1, y1],
                        "crop": str(crop_path)
                    })

            vis_name = f"{Path(pdf_path).stem}_page{idx}_stamp_vis.png"
            img_with_boxes.save(output_dir / vis_name)
            results["pages"].append(page_info)

        meta_path = output_dir / f"{Path(pdf_path).stem}_stamps.json"
        with open(meta_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ Detected {sum(len(p['detections']) for p in results['pages'])} stamp(s)")
        return {"meta": str(meta_path), "out_dir": str(output_dir)}

    # ============ SIGNATURE DETECTION (TILED) ============
    def detect_signatures_tiled(self, pdf_path, output_dir, tile_size=512, overlap=0.2):
        """Detect signatures using tiled approach for better detection with black backgrounds"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images = self.pdf_to_images(pdf_path)
        all_page_results = []

        for page_num, page_img in enumerate(images, start=1):
            # Save page as temporary image
            img_path = output_dir / f"{Path(pdf_path).stem}_page{page_num}.jpg"
            page_img.save(img_path)

            # Read with OpenCV and preprocess
            img = cv2.imread(str(img_path))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rgb_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            h, w = rgb_img.shape[:2]
            step = int(tile_size * (1 - overlap))

            all_boxes, all_confs = [], []

            # Slide window across the full image (tiling approach)
            for y in range(0, h, step):
                for x in range(0, w, step):
                    y_end = min(y + tile_size, h)
                    x_end = min(x + tile_size, w)
                    tile = rgb_img[y:y_end, x:x_end]

                    preds = self.signature_model(source=tile, conf=0.05, imgsz=1024, iou=0.5, verbose=False)
                    dets = preds[0].boxes

                    if len(dets) == 0:
                        continue

                    for box, conf in zip(dets.xyxy.cpu().numpy(), dets.conf.cpu().numpy()):
                        x1, y1, x2, y2 = box
                        all_boxes.append([x1 + x, y1 + y, x2 + x, y2 + y])
                        all_confs.append(conf)

            if not all_boxes:
                all_page_results.append({"page": page_num, "signatures": []})
                continue

            # Apply NMS
            boxes = torch.tensor(all_boxes, dtype=torch.float32)
            scores = torch.tensor(all_confs, dtype=torch.float32)
            keep_indices = torch.ops.torchvision.nms(boxes, scores, iou_threshold=0.5)
            
            final_boxes = boxes[keep_indices]
            final_scores = scores[keep_indices]

            page_signatures = []
            for sig_idx, (box, conf) in enumerate(zip(final_boxes, final_scores), start=1):
                x1, y1, x2, y2 = [int(coord) for coord in box.tolist()]
                
                sig_crop = page_img.crop((x1, y1, x2, y2))
                crop_name = f"{Path(pdf_path).stem}_page{page_num}_sig{sig_idx}.png"
                crop_path = output_dir / crop_name
                sig_crop.save(crop_path)

                page_signatures.append({
                    "id": sig_idx,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": float(conf),
                    "crop": str(crop_path)
                })

            self.draw_signatures(page_img, page_signatures, pdf_path, page_num, output_dir)
            all_page_results.append({"page": page_num, "signatures": page_signatures})

        meta_path = output_dir / f"{Path(pdf_path).stem}_signatures.json"
        with open(meta_path, "w") as f:
            json.dump({"pdf": str(pdf_path), "pages": all_page_results}, f, indent=2)

        print(f"  ✓ Detected {sum(len(p['signatures']) for p in all_page_results)} signature(s)")
        return {"meta": str(meta_path), "out_dir": str(output_dir)}

    def draw_signatures(self, image, signatures, pdf_path, page_num, output_dir):
        """Draw bounding boxes for signatures"""
        img_with_boxes = image.copy()
        draw = ImageDraw.Draw(img_with_boxes)
        
        for sig in signatures:
            bbox = sig["bbox"]
            conf = sig["confidence"]
            draw.rectangle(bbox, outline="blue", width=3)
            draw.text((bbox[0], bbox[1] - 15), f"Sig: {conf:.2f}", fill="blue")

        vis_name = f"{Path(pdf_path).stem}_page{page_num}_sig_vis.png"
        vis_path = output_dir / vis_name
        img_with_boxes.save(vis_path)

    # ============ PHOTO DETECTION ============
    # def detect_photos(self, pdf_path, output_dir, score_thresh=0.5):
    #     """Detect photos/faces in PDF using YOLO"""
    #     output_dir = Path(output_dir)
    #     output_dir.mkdir(parents=True, exist_ok=True)
        
    #     images = self.pdf_to_images(pdf_path)
    #     results = {"pdf": str(pdf_path), "pages": []}

    #     for idx, img in enumerate(images, start=1):
    #         img_path = output_dir / f"{Path(pdf_path).stem}_page{idx}.jpg"
    #         img.save(img_path)

    #         # Use YOLO for detection
    #         preds = self.photo_model.predict(
    #             source=str(img_path), 
    #             conf=score_thresh, 
    #             save=False, 
    #             verbose=False
    #         )

    #         page_info = {"page": idx, "detections": []}
            
    #         if len(preds) > 0 and len(preds[0].boxes) > 0:
    #             img_with_boxes = img.copy()
    #             draw = ImageDraw.Draw(img_with_boxes)
                
    #             for det_i, box in enumerate(preds[0].boxes, start=1):
    #                 x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
    #                 conf = float(box.conf[0])
                    
    #                 if conf > score_thresh:
    #                     photo_crop = img.crop((x1, y1, x2, y2))
    #                     crop_name = f"{Path(pdf_path).stem}_page{idx}_photo{det_i}.png"
    #                     crop_path = output_dir / crop_name
    #                     photo_crop.save(crop_path)

    #                     draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
    #                     draw.text((x1, y1 - 15), f"Photo: {conf:.2f}", fill="green")

    #                     page_info["detections"].append({
    #                         "id": det_i,
    #                         "confidence": conf,
    #                         "box": [x1, y1, x2, y2],
    #                         "crop": str(crop_path)
    #                     })

    #             vis_name = f"{Path(pdf_path).stem}_page{idx}_photo_vis.png"
    #             img_with_boxes.save(output_dir / vis_name)

    #         results["pages"].append(page_info)

    #     meta_path = output_dir / f"{Path(pdf_path).stem}_photos.json"
    #     with open(meta_path, "w") as f:
    #         json.dump(results, f, indent=2)

    #     print(f"  ✓ Detected {sum(len(p['detections']) for p in results['pages'])} photo(s)")
    #     return {"meta": str(meta_path), "out_dir": str(output_dir)}
    
    ## this below code is - photo detection- converted to grayscale concept
    def detect_photos(self, pdf_path, output_dir, score_thresh=0.5):
        """Detect photos/faces in PDF using YOLO with grayscale preprocessing"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        images = self.pdf_to_images(pdf_path)
        results = {"pdf": str(pdf_path), "pages": []}

        for idx, img in enumerate(images, start=1):
            # Convert to grayscale for better photo detection
            gray_images = img.convert("L")
            
            img_path = output_dir / f"{Path(pdf_path).stem}_page{idx}.jpg"
            gray_images.save(img_path)

            # Use YOLO with lower confidence and higher resolution
            preds = self.photo_model(
                str(img_path), 
                conf=0.05,  # Lower confidence to catch more candidates
                imgsz=1024,  # Higher resolution for better detection
                iou=0.5,
                verbose=False
            )

            page_info = {"page": idx, "detections": []}
            
            if len(preds) > 0 and len(preds[0].boxes) > 0:
                img_with_boxes = img.copy()  # Use original RGB for visualization
                draw = ImageDraw.Draw(img_with_boxes)
                
                for det_i, box in enumerate(preds[0].boxes, start=1):
                    x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    
                    if conf > score_thresh:
                        # Crop from original RGB image
                        photo_crop = img.crop((x1, y1, x2, y2))
                        crop_name = f"{Path(pdf_path).stem}_page{idx}_photo{det_i}.png"
                        crop_path = output_dir / crop_name
                        photo_crop.save(crop_path)

                        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
                        draw.text((x1, y1 - 15), f"Photo: {conf:.2f}", fill="green")

                        page_info["detections"].append({
                            "id": det_i,
                            "confidence": conf,
                            "box": [x1, y1, x2, y2],
                            "crop": str(crop_path)
                        })

                vis_name = f"{Path(pdf_path).stem}_page{idx}_photo_vis.png"
                img_with_boxes.save(output_dir / vis_name)

            results["pages"].append(page_info)

        meta_path = output_dir / f"{Path(pdf_path).stem}_photos.json"
        with open(meta_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"  ✓ Detected {sum(len(p['detections']) for p in results['pages'])} photo(s)")
        return {"meta": str(meta_path), "out_dir": str(output_dir)}


    # ============ OCR DETECTION ============
    def detect_ocr(self, pdf_path, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔍 Starting OCR extraction for: {Path(pdf_path).name}")
        try:
            # Call the already integrated Mistral OCR process
            ocr_result = self.process_pdf_with_mistral(pdf_path, output_dir)

            if ocr_result is None:
                print("❌ OCR extraction failed.")
                return {"status": "failed", "pdf": str(pdf_path), "output_dir": str(output_dir)}

            # Save a lightweight summary meta file for UI consumption
            summary_meta = {
                "pdf": str(pdf_path),
                "output_dir": str(output_dir),
                "status": "success",
                "pages_processed": ocr_result.get("usage_info", {}).get("pages_processed", "unknown"),
                "output_json": str(output_dir / f"{Path(pdf_path).stem}_ocr.json")
            }

            summary_path = output_dir / f"{Path(pdf_path).stem}_ocr_meta.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary_meta, f, indent=2, ensure_ascii=False)

            print(f"  ✓ OCR extraction completed successfully for {Path(pdf_path).name}")
            return summary_meta

        except Exception as e:
            print(f"❌ Error during OCR extraction: {e}")
            return {"status": "error", "pdf": str(pdf_path), "error": str(e)}

    
    def encode_pdf_to_base64(self,pdf_path):
        """Read a PDF file and encode it to base64"""
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_content = pdf_file.read()
                base64_encoded = base64.b64encode(pdf_content).decode('utf-8')
                return base64_encoded
        except FileNotFoundError:
            print(f"❌ Error: PDF file not found at {pdf_path}")
            return None
        except Exception as e:
            print(f"❌ Error reading PDF file: {e}")
            return None

    def process_pdf_with_mistral(self,pdf_path, output_path):
        """Process a local PDF file using Mistral Document AI"""
        
        print("🔄 DEMO: PDF to Structured Data with Mistral Document AI")
        print("=" * 60)
        
        load_dotenv()
        # Get API key from environment
        api_key = os.getenv('FOUNDRY_KEY')
        
    
        if not api_key:
            print("❌ Error: AZURE_API_KEY not found in environment variables")
            return None
        
        # Show file info
        file_size = os.path.getsize(pdf_path) / 1024  # KB
        print(f"📄 Input File: {pdf_path}")
        print(f"📊 File Size: {file_size:.1f} KB")
        
        # Encode PDF to base64
        print(f"\n🔄 Step 1: Encoding PDF to base64...")
        time.sleep(1)  # Demo pause
        base64_content = self.encode_pdf_to_base64(pdf_path)
        if not base64_content:
            return None
        
        print(f"✅ Encoded {len(base64_content):,} characters")
        
        # API endpoint and headers
        # azure_endpoint = os.getenv('AZURE_ENDPOINT')
        azure_endpoint = os.getenv('OCR_FOUNDRY_ENDPOINT')
        
        
        url = azure_endpoint
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Request payload
        model_name = 'mistral-document-ai-2505'
        payload = {
            "model": model_name,
            "document": {
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{base64_content}"
            },
            "include_image_base64": True
        }
        
        try:
            print(f"\n🚀 Step 2: Sending to Mistral Document AI...")
            print(f"🔗 Endpoint: {url}")
            time.sleep(1)  # Demo pause
            
            start_time = time.time()
            response = requests.post(url, headers=headers, json=payload)
            end_time = time.time()
            
            response.raise_for_status()
            
            result = response.json()
            processing_time = end_time - start_time
            
            print(f"✅ Success! Processing completed in {processing_time:.2f} seconds")
            print(f"📄 Pages processed: {result['usage_info']['pages_processed']}")
            print(f"📊 Document size: {result['usage_info']['doc_size_bytes']:,} bytes")
            
            output_file = f"{output_path}/{Path(pdf_path).stem}_ocr.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            return result
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error making API request: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            return None

    # def detect_ocr(self, pdf_path, output_dir):
    #     """Extract structured data from PDF using Mistral OCR"""
    #     output_dir = Path(output_dir)
    #     output_dir.mkdir(parents=True, exist_ok=True)

    #     # Convert PDF to base64
    #     with open(pdf_path, "rb") as file:
    #         file_bytes = file.read()
    #         encoded_bytes = base64.b64encode(file_bytes).decode("utf-8")

    #     try:
    #         annotations_response = self.ocr_client.ocr.process(
    #             model="mistral-ocr-latest",
    #             pages=list(range(8)),
    #             document={
    #                 "type": "document_url",
    #                 "document_url": f"data:application/pdf;base64,{encoded_bytes}"
    #             },
    #             document_annotation_format=response_format_from_pydantic_model(Document),
    #             include_image_base64=False
    #         )

    #         extracted_data = json.loads(annotations_response.document_annotation)
            
    #         # Save extracted data
    #         output_file = output_dir / f"{Path(pdf_path).stem}_ocr.json"
    #         with open(output_file, "w", encoding="utf-8") as f:
    #             json.dump(extracted_data, f, indent=2, ensure_ascii=False)

    #         print(f"  ✓ OCR extraction completed")
    #         return {"data": extracted_data, "output_file": str(output_file)}

    #     except Exception as e:
    #         print(f"  ✗ OCR extraction failed: {str(e)}")
    #         return {"error": str(e)}

    # ============ PROCESS DIRECTORY ============
    def process_pdf_directory(self, main_dir, detection_type="all", score_thresh=0.7):
        """Process all PDFs in directory for specified detection type(s)"""
        main_dir = Path(main_dir)
        pdf_files = list(main_dir.rglob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {main_dir}")
            return

        print(f"\n{'='*60}")
        print(f"Processing {len(pdf_files)} PDF(s) for {detection_type.upper()} detection")
        print(f"{'='*60}\n")

        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            try:
                if detection_type in ["stamp", "all"]:
                    output_dir = main_dir / "stamp_outputs"
                    self.detect_stamps(str(pdf_file), str(output_dir), score_thresh=score_thresh)
                
                if detection_type in ["signature", "all"]:
                    output_dir = main_dir / "signature_outputs"
                    self.detect_signatures_tiled(str(pdf_file), str(output_dir))
                
                if detection_type in ["photo", "all"]:
                    output_dir = main_dir / "photo_outputs"
                    self.detect_photos(str(pdf_file), str(output_dir), score_thresh=score_thresh)
                
                if detection_type in ["ocr", "all"]:
                    output_dir = main_dir / "ocr_outputs"
                    
                    '''This is the old way to leverage OCR, we moved to foundry now'''
                    # self.detect_ocr(str(pdf_file), str(output_dir)) 
                    self.detect_ocr(str(pdf_file), str(output_dir))
                    
                
                print(f"  ✓ Completed\n")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue

        print(f"{'='*60}")
        print(f"All detections completed!")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Initialize detector
    detector = ObjectDetection()
    
    # Process all detection types
    # Options: "stamp", "signature", "photo", "ocr", "all"
    
    detector.process_pdf_directory("data", detection_type="all", score_thresh=0.7)
    
    # Or run individually:
    # detector.process_pdf_directory("data", detection_type="stamp", score_thresh=0.7)
    # detector.process_pdf_directory("data", detection_type="signature")
    # detector.process_pdf_directory("data", detection_type="photo")
    # detector.process_pdf_directory("data", detection_type="ocr")