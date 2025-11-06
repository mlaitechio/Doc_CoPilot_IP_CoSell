import streamlit as st
import os
import json
import shutil
from pathlib import Path
import zipfile
from main import ObjectDetection

st.set_page_config(
    page_title="Document Detection System",
    page_icon="📄",
    layout="wide"
)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
    st.session_state.processed = False
    st.session_state.results_dir = None
    st.session_state.last_upload_count = 0

def initialize_detector():
    """Initialize the detection model (cached)"""
    if st.session_state.detector is None:
        with st.spinner("Loading detection models... This may take a minute."):
            st.session_state.detector = ObjectDetection()
        st.success("✓ Models loaded successfully!")

def clear_previous_results(temp_dir):
    """Clear all previous processing results"""
    temp_path = Path(temp_dir)
    
    if temp_path.exists():
        # Remove all subdirectories and files
        shutil.rmtree(temp_path)
        st.info("🗑️ Cleared previous results")
    
    # Recreate empty directory
    temp_path.mkdir(exist_ok=True)
    
    # Reset session state
    st.session_state.processed = False
    st.session_state.results_dir = None

def create_zip_from_folder(folder_path, zip_name):
    """Create a ZIP file from folder contents"""
    zip_path = Path(folder_path).parent / zip_name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    return zip_path

def display_results(output_dir, detection_type):
    """Display detection results"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        st.warning(f"No {detection_type} results found.")
        return
    
    # Get all files
    image_files = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))
    json_files = list(output_path.glob("*.json"))
    
    st.subheader(f"📊 {detection_type.upper()} Results")
    
    # Display JSON metadata
    if json_files:
        with st.expander(f"📄 View {detection_type.upper()} Metadata"):
            for json_file in json_files:
                st.markdown(f"**{json_file.name}**")
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    st.json(data)
    
    # Display images
    if image_files:
        st.markdown(f"**Detected {detection_type.upper()}s:**")
        
        # Filter: Show only crops, not visualization images
        crop_images = [img for img in image_files if 'vis' not in img.name]
        vis_images = [img for img in image_files if 'vis' in img.name]
        
        # Show visualization images
        if vis_images:
            st.markdown("**📸 Visualizations (with bounding boxes):**")
            cols = st.columns(min(3, len(vis_images)))
            for idx, img_path in enumerate(vis_images):
                with cols[idx % 3]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
        
        # Show extracted crops
        if crop_images:
            st.markdown(f"**✂️ Extracted {detection_type.upper()}s:**")
            cols = st.columns(min(4, len(crop_images)))
            for idx, img_path in enumerate(crop_images):
                with cols[idx % 4]:
                    st.image(str(img_path), caption=img_path.name, use_container_width=True)
    
    # Download button for this detection type
    if image_files or json_files:
        zip_path = create_zip_from_folder(output_path, f"{detection_type}_results.zip")
        with open(zip_path, "rb") as f:
            st.download_button(
                label=f"⬇️ Download {detection_type.upper()} Results (ZIP)",
                data=f,
                file_name=f"{detection_type}_results.zip",
                mime="application/zip"
            )

# ============ MAIN UI ============

st.title("📄 Document Detection System")
st.markdown("**Extract Stamps, Signatures, Photos, and OCR data from PDFs**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Detection options
    st.subheader("Select Detection Types:")
    detect_stamp = st.checkbox("🔖 Stamps", value=True)
    detect_signature = st.checkbox("✍️ Signatures", value=True)
    detect_photo = st.checkbox("📷 Photos", value=True)
    detect_ocr = st.checkbox("📝 OCR (Text Extraction)", value=False)
    
    st.divider()
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        score_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Higher values = fewer but more confident detections"
        )
        
        tile_size = st.slider(
            "Signature Tile Size (pixels)",
            min_value=256,
            max_value=1024,
            value=512,
            step=128,
            help="Tile size for signature detection. Larger = slower but better for big documents"
        )
        
        overlap = st.slider(
            "Signature Tile Overlap",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Overlap between tiles (0.2 = 20% overlap)"
        )

# Main content
uploaded_files = st.file_uploader(
    "📤 Upload PDF Files",
    type=['pdf'],
    accept_multiple_files=True,
    help="You can upload multiple PDF files at once"
)

if uploaded_files:
    # Check if new files were uploaded (different from last time)
    current_upload_count = len(uploaded_files)
    
    if current_upload_count != st.session_state.last_upload_count:
        # New upload detected - clear previous results
        temp_dir = Path("temp_uploads")
        if temp_dir.exists():
            clear_previous_results(temp_dir)
        st.session_state.last_upload_count = current_upload_count
    
    st.success(f"✓ {len(uploaded_files)} file(s) uploaded successfully!")
    
    # Create temporary directory for uploads
    temp_dir = Path("temp_uploads")
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded files
    saved_files = []
    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append(file_path)
        st.info(f"📁 Saved: {uploaded_file.name}")
    
    # Process button
    col1, col2 = st.columns([3, 1])
    
    with col1:
        process_button = st.button("🚀 Start Processing", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            clear_previous_results(temp_dir)
            st.rerun()
    
    if process_button:
        # Determine detection types
        detection_types = []
        if detect_stamp:
            detection_types.append("stamp")
        if detect_signature:
            detection_types.append("signature")
        if detect_photo:
            detection_types.append("photo")
        if detect_ocr:
            detection_types.append("ocr")
        
        if not detection_types:
            st.error("⚠️ Please select at least one detection type!")
        else:
            # Initialize detector
            initialize_detector()
            
            # Process each file
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, file_path in enumerate(saved_files):
                status_text.text(f"Processing {file_path.name}...")
                
                try:
                    # Process based on selected detection types
                    for detection_type in detection_types:
                        output_dir = temp_dir / f"{detection_type}_outputs"
                        
                        if detection_type == "stamp":
                            st.session_state.detector.detect_stamps(
                                str(file_path),
                                str(output_dir),
                                score_thresh=score_threshold
                            )
                        
                        elif detection_type == "signature":
                            st.session_state.detector.detect_signatures_tiled(
                                str(file_path),
                                str(output_dir),
                                tile_size=tile_size,
                                overlap=overlap
                            )
                        
                        elif detection_type == "photo":
                            st.session_state.detector.detect_photos(
                                str(file_path),
                                str(output_dir),
                                score_thresh=score_threshold
                            )
                        
                        elif detection_type == "ocr":
                            st.session_state.detector.detect_ocr(
                                str(file_path),
                                str(output_dir)
                            )
                    
                    st.success(f"✓ Completed: {file_path.name}")
                
                except Exception as e:
                    st.error(f"✗ Error processing {file_path.name}: {str(e)}")
                
                # Update progress
                progress_bar.progress((idx + 1) / len(saved_files))
            
            status_text.text("✓ All files processed!")
            st.session_state.processed = True
            st.session_state.results_dir = temp_dir
            
            # Celebrate!
            # st.balloons()

# Display results
if st.session_state.processed and st.session_state.results_dir:
    st.divider()
    st.header("📊 Detection Results")
    
    # Create tabs for each detection type
    tabs = []
    if detect_stamp:
        tabs.append("🔖 Stamps")
    if detect_signature:
        tabs.append("✍️ Signatures")
    if detect_photo:
        tabs.append("📷 Photos")
    if detect_ocr:
        tabs.append("📝 OCR")
    
    if tabs:
        tab_objects = st.tabs(tabs)
        
        tab_idx = 0
        if detect_stamp:
            with tab_objects[tab_idx]:
                display_results(
                    st.session_state.results_dir / "stamp_outputs",
                    "stamp"
                )
            tab_idx += 1
        
        if detect_signature:
            with tab_objects[tab_idx]:
                display_results(
                    st.session_state.results_dir / "signature_outputs",
                    "signature"
                )
            tab_idx += 1
        
        if detect_photo:
            with tab_objects[tab_idx]:
                display_results(
                    st.session_state.results_dir / "photo_outputs",
                    "photo"
                )
            tab_idx += 1
        
        if detect_ocr:
            with tab_objects[tab_idx]:
                display_results(
                    st.session_state.results_dir / "ocr_outputs",
                    "ocr"
                )
    
    # Download all results
    st.divider()
    st.subheader("📦 Download All Results")
    
    if st.button("📥 Create Complete Results Package", use_container_width=True):
        with st.spinner("Creating ZIP archive..."):
            all_results_zip = create_zip_from_folder(
                st.session_state.results_dir,
                "all_detection_results.zip"
            )
            
            with open(all_results_zip, "rb") as f:
                st.download_button(
                    label="⬇️ Download Complete Results (ZIP)",
                    data=f,
                    file_name="all_detection_results.zip",
                    mime="application/zip",
                    type="primary",
                    use_container_width=True
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Supports: Stamp Detection • Signature Detection • Photo Detection • OCR Extraction</p>
</div>
""", unsafe_allow_html=True)