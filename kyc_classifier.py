#kyc_classifier.py
import torch
import cv2
import numpy as np
import json
from PIL import Image
from pathlib import Path
from transformers import (
    ViTImageProcessor,
    AutoModelForImageClassification
)

try:
    import fitz  # PyMuPDF
except ImportError:
    raise RuntimeError("PyMuPDF not installed. Run: pip install pymupdf")


class KYCClassificationQuality:
    
    def __init__(self):
        print("Loading KYC Classification and Quality Check models...")
        
        # Document Classification Model (Microsoft DIT on RVL-CDIP dataset)
        self.classification_processor = ViTImageProcessor.from_pretrained(
            "microsoft/dit-base-finetuned-rvlcdip"
        )
        self.classification_model = AutoModelForImageClassification.from_pretrained(
            "microsoft/dit-base-finetuned-rvlcdip"
        )
        
        print("✓ Models loaded successfully\n")
    
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
    
    # ============ DOCUMENT CLASSIFICATION ============
    
    def classify_document(self, image):
        
        inputs = self.classification_processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.classification_model(**inputs)
            logits = outputs.logits
        
        predicted_class_idx = logits.argmax(-1).item()
        confidence = torch.softmax(logits, dim=1).max().item()
        
        doc_type = self.classification_model.config.id2label[predicted_class_idx]
        
        return {
            "document_type": doc_type,
            "confidence": float(confidence),
            "class_id": predicted_class_idx
        }
    
    def map_to_kyc_category(self, doc_type):
        
        doc_type_lower = doc_type.lower()
        
        kyc_mapping = {
            "letter": "Official Letter / Communication",
            "form": "KYC Application Form",
            "invoice": "Proof of Address (Utility Bill)",
            "advertisement": "Not KYC Document",
            "budget": "Financial Statement",
            "email": "Email Communication",
            "handwritten": "Handwritten Form/Application",
            "memo": "Official Memo/Notice",
            "news article": "Not KYC Document",
            "presentation": "Not KYC Document",
            "questionnaire": "Survey/Questionnaire",
            "resume": "Resume/CV",
            "scientific publication": "Not KYC Document",
            "scientific report": "Report/Statement",
            "specification": "Not KYC Document",
            "file folder": "Mixed Documents"
        }
        
        return kyc_mapping.get(doc_type_lower, "Unknown Document Type")
    
    def classify_kyc_document(self, image):
        
        base_result = self.classify_document(image)
        kyc_category = self.map_to_kyc_category(base_result["document_type"])
        is_kyc = "Not KYC Document" not in kyc_category and "Unknown" not in kyc_category
        
        return {
            "base_type": base_result["document_type"],
            "kyc_category": kyc_category,
            "confidence": base_result["confidence"],
            "is_valid_kyc": is_kyc
        }
    
    # ============ DOCUMENT QUALITY CHECK ============
    
    def check_resolution(self, img):
        """Check if image resolution is adequate"""
        height, width = img.shape[:2]
        total_pixels = height * width
        
        if total_pixels < 500000:
            status = "Low"
            score = 0.3
        elif total_pixels < 2000000:
            status = "Acceptable"
            score = 0.7
        else:
            status = "Good"
            score = 1.0
        
        return {
            "width": width,
            "height": height,
            "megapixels": round(total_pixels / 1000000, 2),
            "status": status,
            "score": score
        }
    
    def check_brightness(self, gray):
        """Check if image brightness is optimal"""
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 80:
            status = "Too Dark"
            score = 0.4
        elif mean_brightness > 180:
            status = "Too Bright"
            score = 0.5
        elif 100 <= mean_brightness <= 150:
            status = "Optimal"
            score = 1.0
        else:
            status = "Acceptable"
            score = 0.8
        
        return {
            "mean_value": float(mean_brightness),
            "status": status,
            "score": score
        }
    
    def check_contrast(self, gray):
        """Check image contrast quality"""
        std_dev = np.std(gray)
        
        if std_dev < 30:
            status = "Low Contrast"
            score = 0.4
        elif std_dev < 50:
            status = "Acceptable"
            score = 0.7
        else:
            status = "Good Contrast"
            score = 1.0
        
        return {
            "std_deviation": float(std_dev),
            "status": status,
            "score": score
        }
    
    def check_sharpness(self, gray):
        """Detect blur using Laplacian variance"""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            status = "Blurry"
            score = 0.3
        elif laplacian_var < 300:
            status = "Acceptable Sharpness"
            score = 0.7
        else:
            status = "Sharp"
            score = 1.0
        
        return {
            "laplacian_variance": float(laplacian_var),
            "status": status,
            "score": score
        }
    
    def check_noise(self, gray):
        """Estimate image noise level"""
        median = cv2.medianBlur(gray, 5)
        noise = np.mean(np.abs(gray.astype(float) - median.astype(float)))
        
        if noise > 20:
            status = "High Noise"
            score = 0.4
        elif noise > 10:
            status = "Moderate Noise"
            score = 0.7
        else:
            status = "Low Noise"
            score = 1.0
        
        return {
            "noise_level": float(noise),
            "status": status,
            "score": score
        }
    
    def check_skew(self, gray):
        """Detect document skew/rotation angle"""
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return {
                "angle": 0.0,
                "status": "Cannot Detect",
                "score": 0.5
            }
        
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        median_angle = np.median(angles) if angles else 0
        
        if abs(median_angle) < 2:
            status = "Well Aligned"
            score = 1.0
        elif abs(median_angle) < 5:
            status = "Slightly Skewed"
            score = 0.8
        else:
            status = "Skewed"
            score = 0.5
        
        return {
            "angle": float(median_angle),
            "status": status,
            "score": score
        }
    
    def assess_quality(self, image_path):
        
        img = cv2.imread(str(image_path))
        if img is None:
            return {"error": "Could not read image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        quality_metrics = {
            "resolution": self.check_resolution(img),
            "brightness": self.check_brightness(gray),
            "contrast": self.check_contrast(gray),
            "sharpness": self.check_sharpness(gray),
            "noise": self.check_noise(gray),
            "skew": self.check_skew(gray)
        }
        
        weights = {
            "resolution": 0.20,
            "brightness": 0.15,
            "contrast": 0.15,
            "sharpness": 0.30,
            "noise": 0.10,
            "skew": 0.10
        }
        
        overall_score = sum(
            quality_metrics[metric]["score"] * weight 
            for metric, weight in weights.items()
        )
        
        if overall_score >= 0.8:
            quality_label = "Excellent"
            recommendation = "Document is ready for processing."
        elif overall_score >= 0.6:
            quality_label = "Good"
            recommendation = "Document quality is good. Can proceed with processing."
        elif overall_score >= 0.4:
            quality_label = "Acceptable"
            recommendation = "Quality is acceptable but may affect accuracy. Consider re-scanning if possible."
        else:
            quality_label = "Poor"
            recommendation = "Quality is poor. Please re-scan the document for better results."
        
        quality_metrics["overall_score"] = round(overall_score, 2)
        quality_metrics["quality_label"] = quality_label
        quality_metrics["recommendation"] = recommendation
        
        return quality_metrics
    
    # ============ PROCESS PDF DOCUMENTS - UPDATED WITH SEPARATE OUTPUTS ============
    
    def process_pdf(self, pdf_path, classification_dir="classification_outputs/kyc_classification",
                   quality_dir="data_quality/doc_quality", *args, **kwargs):
        
        # Create output directories
        classification_dir = Path(classification_dir)
        quality_dir = Path(quality_dir)
        classification_dir.mkdir(parents=True, exist_ok=True)
        quality_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_path = Path(pdf_path)
        images = self.pdf_to_images(pdf_path)
        
        # Separate results structures
        classification_results = {
            "pdf": str(pdf_path),
            "total_pages": len(images),
            "pages": []
        }
        
        quality_results = {
            "pdf": str(pdf_path),
            "total_pages": len(images),
            "pages": []
        }
        
        # Temporary directory for image conversion
        temp_dir = Path("temp_images")
        temp_dir.mkdir(exist_ok=True)
        
        for page_num, img in enumerate(images, start=1):
            print(f"Processing page {page_num}/{len(images)}...")
            
            # Save temporary image for quality check
            temp_img_path = temp_dir / f"temp_page_{page_num}.jpg"
            img.save(temp_img_path)
            
            # Classification
            classification_result = self.classify_kyc_document(img)
            classification_results["pages"].append({
                "page": page_num,
                "classification": classification_result
            })
            
            # Quality check
            quality_result = self.assess_quality(temp_img_path)
            quality_results["pages"].append({
                "page": page_num,
                "quality": quality_result
            })
            
            # Print summary
            print(f"  Type: {classification_result['kyc_category']}")
            print(f"  Confidence: {classification_result['confidence']:.2%}")
            print(f"  Quality: {quality_result['quality_label']} ({quality_result['overall_score']:.2f})")
            print()
            
            # Clean up temp file
            temp_img_path.unlink()
        
        # Clean up temp directory
        temp_dir.rmdir()
        
        # Save classification results
        classification_path = classification_dir / f"{pdf_path.stem}_classification.json"
        with open(classification_path, "w", encoding="utf-8") as f:
            json.dump(classification_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Classification results saved to: {classification_path}")
        
        # Save quality results
        quality_path = quality_dir / f"{pdf_path.stem}_quality.json"
        with open(quality_path, "w", encoding="utf-8") as f:
            json.dump(quality_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Quality results saved to: {quality_path}\n")
        
        return classification_results, quality_results
    
    def process_directory(self, data_dir="data", 
                         classification_dir="classification_outputs/kyc_classification",
                         quality_dir="data_quality/doc_quality"):
        
        data_dir = Path(data_dir)
        classification_dir = Path(classification_dir)
        quality_dir = Path(quality_dir)
        
        classification_dir.mkdir(parents=True, exist_ok=True)
        quality_dir.mkdir(parents=True, exist_ok=True)
        
        pdf_files = list(data_dir.rglob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {data_dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing {len(pdf_files)} PDF(s) for Classification & Quality Check")
        print(f"{'='*60}\n")
        
        for pdf_file in pdf_files:
            print(f"Processing: {pdf_file.name}")
            
            try:
                self.process_pdf(
                    str(pdf_file),
                    classification_dir=str(classification_dir),
                    quality_dir=str(quality_dir)
                )
                print(f"  ✓ Completed\n")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}\n")
                continue
        
        print(f"{'='*60}")
        print(f"All processing completed!")
        print(f"Classification results: {classification_dir}")
        print(f"Quality results: {quality_dir}")
        print(f"{'='*60}\n")


# ============ MAIN EXECUTION ============

if __name__ == "__main__":
    print("="*60)
    print("KYC Document Classification & Quality Check")
    print("="*60)
    print()
    
    classifier = KYCClassificationQuality()
    
    # Process ALL PDFs with separate output directories
    classifier.process_directory(
        data_dir="data",
        classification_dir="classification_outputs/kyc_classification",
        quality_dir="data_quality/doc_quality"
    )
    
    print("="*60)
    print("Processing Complete!")
    print("="*60)
    print()
    print("Results saved to:")
    print("  - Classification: classification_outputs/kyc_classification/")
    print("  - Quality: data_quality/doc_quality/")
    print()