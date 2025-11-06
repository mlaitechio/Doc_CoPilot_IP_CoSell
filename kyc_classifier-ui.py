import streamlit as st
import os
import json
import shutil
from pathlib import Path
from kyc_classifier import KYCClassificationQuality

st.set_page_config(
    page_title="KYC Document Analyzer",
    page_icon="📋",
    layout="wide"
)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
    st.session_state.processed = False
    st.session_state.classification_results = None
    st.session_state.quality_results = None
    st.session_state.last_upload_count = 0

def initialize_classifier():
    """Initialize the KYC classifier (cached)"""
    if st.session_state.classifier is None:
        with st.spinner("Loading KYC Classification and Quality Check models..."):
            st.session_state.classifier = KYCClassificationQuality()
        st.success("✓ Models loaded successfully!")

def clear_previous_results():
    """Clear all previous processing results"""
    temp_dir = Path("temp_kyc_uploads")
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
        st.info("🗑️ Cleared previous results")
    
    temp_dir.mkdir(exist_ok=True)
    
    # Reset session state
    st.session_state.processed = False
    st.session_state.classification_results = None
    st.session_state.quality_results = None

def display_classification_results(classification_dir):
    """Display classification results"""
    classification_path = Path(classification_dir)
    
    if not classification_path.exists():
        st.warning("No classification results found.")
        return
    
    json_files = list(classification_path.glob("*.json"))
    
    if not json_files:
        st.warning("No classification results found.")
        return
    
    st.subheader("📊 Document Classification Results")
    
    for json_file in json_files:
        with st.expander(f"📄 {json_file.stem}", expanded=True):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            st.markdown(f"**PDF:** `{data['pdf']}`")
            st.markdown(f"**Total Pages:** {data['total_pages']}")
            
            for page_data in data['pages']:
                page_num = page_data['page']
                classification = page_data['classification']
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Page", page_num)
                
                with col2:
                    st.metric(
                        "Document Type", 
                        classification['kyc_category'],
                        help=f"Base Type: {classification['base_type']}"
                    )
                
                with col3:
                    confidence_pct = f"{classification['confidence']*100:.1f}%"
                    st.metric("Confidence", confidence_pct)
                
                # Valid KYC indicator
                if classification['is_valid_kyc']:
                    st.success("✅ Valid KYC Document")
                else:
                    st.warning("⚠️ Not a KYC Document")
                
                st.divider()

def display_quality_results(quality_dir):
    """Display quality check results"""
    quality_path = Path(quality_dir)
    
    if not quality_path.exists():
        st.warning("No quality results found.")
        return
    
    json_files = list(quality_path.glob("*.json"))
    
    if not json_files:
        st.warning("No quality results found.")
        return
    
    st.subheader("🔍 Document Quality Assessment")
    
    for json_file in json_files:
        with st.expander(f"📄 {json_file.stem}", expanded=True):
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            st.markdown(f"**PDF:** `{data['pdf']}`")
            st.markdown(f"**Total Pages:** {data['total_pages']}")
            
            for page_data in data['pages']:
                page_num = page_data['page']
                quality = page_data['quality']
                
                st.markdown(f"### Page {page_num}")
                
                # Overall quality
                overall_score = quality['overall_score']
                quality_label = quality['quality_label']
                
                # Color code based on quality
                if overall_score >= 0.8:
                    score_color = "🟢"
                elif overall_score >= 0.6:
                    score_color = "🟡"
                else:
                    score_color = "🔴"
                
                st.markdown(f"### {score_color} Overall Quality: **{quality_label}** ({overall_score:.2f})")
                st.info(quality['recommendation'])
                
                # Detailed metrics
                st.markdown("#### Quality Metrics:")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    res = quality['resolution']
                    st.metric(
                        "Resolution", 
                        res['status'],
                        delta=f"{res['megapixels']} MP",
                        help=f"{res['width']}x{res['height']} pixels"
                    )
                    
                    bright = quality['brightness']
                    st.metric(
                        "Brightness", 
                        bright['status'],
                        delta=f"{bright['mean_value']:.0f}/255"
                    )
                
                with col2:
                    contrast = quality['contrast']
                    st.metric(
                        "Contrast", 
                        contrast['status'],
                        delta=f"σ={contrast['std_deviation']:.1f}"
                    )
                    
                    sharp = quality['sharpness']
                    st.metric(
                        "Sharpness", 
                        sharp['status'],
                        delta=f"Var={sharp['laplacian_variance']:.0f}"
                    )
                
                with col3:
                    noise = quality['noise']
                    st.metric(
                        "Noise Level", 
                        noise['status'],
                        delta=f"{noise['noise_level']:.2f}"
                    )
                    
                    skew = quality['skew']
                    st.metric(
                        "Alignment", 
                        skew['status'],
                        delta=f"{skew['angle']:.2f}°"
                    )
                
                # Score breakdown
                st.markdown("#### Score Breakdown:")
                scores = {
                    "Resolution": quality['resolution']['score'],
                    "Brightness": quality['brightness']['score'],
                    "Contrast": quality['contrast']['score'],
                    "Sharpness": quality['sharpness']['score'],
                    "Noise": quality['noise']['score'],
                    "Skew": quality['skew']['score']
                }
                
                # Display as progress bars
                for metric, score in scores.items():
                    st.progress(score, text=f"{metric}: {score:.2f}")
                
                st.divider()

# ============ MAIN UI ============

st.title("📋 KYC Document Analyzer")
st.markdown("**Classify document types and assess quality for KYC processing**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.markdown("### Analysis Options")
    st.info("Both classification and quality check are performed automatically")
    
    st.divider()
    
    st.markdown("### About")
    st.markdown("""
    This tool performs:
    - **Document Classification** - Identifies document type (KYC form, invoice, ID proof, etc.)
    - **Quality Assessment** - Checks resolution, brightness, contrast, sharpness, noise, and alignment
    """)
    
    st.divider()
    

# Main content
uploaded_files = st.file_uploader(
    "📤 Upload PDF Documents for KYC Analysis",
    type=['pdf'],
    accept_multiple_files=True,
    help="Upload one or more PDF documents for classification and quality assessment"
)

if uploaded_files:
    # Check if new files were uploaded
    current_upload_count = len(uploaded_files)
    
    if current_upload_count != st.session_state.last_upload_count:
        clear_previous_results()
        st.session_state.last_upload_count = current_upload_count
    
    st.success(f"✓ {len(uploaded_files)} file(s) uploaded successfully!")
    
    # Create temporary directory for uploads
    temp_dir = Path("temp_kyc_uploads")
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
        process_button = st.button("🚀 Analyze Documents", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            clear_previous_results()
            st.rerun()
    
    if process_button:
        # Initialize classifier
        initialize_classifier()
        
        # Process files
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        classification_dir = temp_dir / "kyc_classify"
        quality_dir = temp_dir / "doc_quality"
        
        for idx, file_path in enumerate(saved_files):
            status_text.text(f"Analyzing {file_path.name}...")
            
            try:
                result = st.session_state.classifier.process_pdf(
                    str(file_path),
                    str(classification_dir),
                    str(quality_dir)
                )
                
                st.success(f"✓ Completed: {file_path.name}")
            
            except Exception as e:
                st.error(f"✗ Error processing {file_path.name}: {str(e)}")
            
            # Update progress
            progress_bar.progress((idx + 1) / len(saved_files))
        
        status_text.text("✓ All files analyzed!")
        st.session_state.processed = True
        st.session_state.classification_results = classification_dir
        st.session_state.quality_results = quality_dir

# Display results
if st.session_state.processed:
    st.divider()
    st.header("📊 Analysis Results")
    
    # Create tabs for results
    tab1, tab2 = st.tabs(["📊 Classification", "🔍 Quality Assessment"])
    
    with tab1:
        display_classification_results(st.session_state.classification_results)
    
    with tab2:
        display_quality_results(st.session_state.quality_results)
    
    # Download options
    st.divider()
    st.subheader("📥 Download Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.classification_results and Path(st.session_state.classification_results).exists():
            # Create ZIP for classification results
            import zipfile
            
            classification_zip = Path("temp_kyc_uploads") / "classification_results.zip"
            with zipfile.ZipFile(classification_zip, 'w') as zipf:
                for file in Path(st.session_state.classification_results).glob("*.json"):
                    zipf.write(file, file.name)
            
            with open(classification_zip, "rb") as f:
                st.download_button(
                    label="📊 Download Classification Results",
                    data=f,
                    file_name="classification_results.zip",
                    mime="application/zip",
                    use_container_width=True
                )
    
    with col2:
        if st.session_state.quality_results and Path(st.session_state.quality_results).exists():
            # Create ZIP for quality results
            quality_zip = Path("temp_kyc_uploads") / "quality_results.zip"
            with zipfile.ZipFile(quality_zip, 'w') as zipf:
                for file in Path(st.session_state.quality_results).glob("*.json"):
                    zipf.write(file, file.name)
            
            with open(quality_zip, "rb") as f:
                st.download_button(
                    label="🔍 Download Quality Results",
                    data=f,
                    file_name="quality_results.zip",
                    mime="application/zip",
                    use_container_width=True
                )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>KYC Document Analyzer</p>
    <p>Classification • Quality Assessment </p>
</div>
""", unsafe_allow_html=True)







