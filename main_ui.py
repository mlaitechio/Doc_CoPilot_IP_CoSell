# Main_UI.py
import streamlit as st
import os
import json
import shutil
import zipfile
from pathlib import Path

# Import module classes
from obj_detect import ObjectDetection
from kyc_classifier import KYCClassificationQuality
from doc_sum_pipeline import Doc_Summarizer

# =========================
# Streamlit Config
# =========================
# =========================
# Custom Page Styling
# =========================
st.set_page_config(
    page_title="Document CoPilot System",
    page_icon="assets/MLAI Digital Logo (1).jpg",
    layout="wide"
)

# st.markdown("""
# <style>
# /* ===== MAIN APP VIEW CONTAINER =====
#    Targets the main area (not sidebar). */
# [data-testid="stAppViewContainer"] > .main {
#     min-height: 100vh;                          /* full viewport */
#     padding-top: 1.5rem;
#     padding-bottom: 1.5rem;
#     background: linear-gradient(135deg, #f5f7fa 0%, #dbe7f3 100%) !important;
#     background-attachment: fixed;
#     background-size: cover;
#     color: #000000;                             /* force dark text */
# }

# /* Sometimes Streamlit nests main content differently, target another wrapper */
# [data-testid="stAppViewContainer"] {
#     background: transparent;                    /* already set above */
# }

# /* Keep the sidebar default look (white) */
# [data-testid="stSidebar"] {
#     background-color: #ffffff;
# }

# /* Make content area cards white and slightly elevated so they stand out on the gradient */
# .css-1d391kg, .stCard, .stExpander {             /* streamlit generated classes vary, but these help */
#     background-color: rgba(255,255,255,0.98) !important;
#     border-radius: 10px;
#     box-shadow: 0 6px 18px rgba(11,20,40,0.06);
# }

# /* File uploader box (improve contrast on gradient) */
# [data-testid="stFileUploader"] > div:first-child {
#     background: rgba(24,24,27,0.9) !important;  /* dark uploader card */
#     color: #fff !important;
#     border-radius: 10px;
#     padding: .75rem;
# }

# /* Make uploader button (inside the card) readable */
# [data-testid="stFileUploader"] button {
#     background: #222 !important;
#     color: #fff !important;
#     border-radius: 8px;
# }

# /* Headers style */
# h1 {
#     color: #0f172a;
#     font-weight: 800;
#     font-size: 2.3rem;
#     margin-bottom: 0.25rem;
# }
# h2, h3 {
#     color: #111827;
# }

# /* Make metrics and other text prominent and dark */
# [data-testid="stMetricValue"], [data-testid="stMetricLabel"], .css-1v3fvcr {
#     color: #0b1220 !important;
# }

# /* Tabs: lighten tab backgrounds to match card look */
# [role="tablist"] > div[role="tab"] {
#     background: rgba(255,255,255,0.9);
#     color: #0b1220;
#     border-radius: 8px;
# }

# /* Progress bar gradient */
# .stProgress > div > div {
#     background: linear-gradient(90deg,#3b82f6 0%,#8b5cf6 100%) !important;
# }

# /* Footer / divider treatment */
# hr {
#     border: none;
#     height: 2px;
#     background: linear-gradient(90deg, transparent, #3b82f6, transparent);
# }

# /* Responsive tweaks: remove big left padding so content aligns nicely */
# main > div.block-container {
#     padding-left: 2.5rem;
#     padding-right: 2.5rem;
# }

# /* If cached styles prevent change, increase specificity using data-testid */
# [data-testid="stAppViewContainer"] .css-1d391kg {
#     background: rgba(255,255,255,0.98) !important;
# }

# /* Ensure links/buttons inside uploader remain readable */
# [data-testid="stFileUploader"] a, [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p {
#     color: #ffffff !important;
# }

# /* small helper: keep sidebar text slightly muted */
# [data-testid="stSidebar"] * {
#     color: #1f2937;
# }
# </style>
# """, unsafe_allow_html=True)



# =========================
# Sidebar Navigation
# =========================
st.sidebar.image("assets\MLAI Digital Logo (1).jpg", use_container_width=True)

st.sidebar.info("Use this unified platform to perform all document analysis tasks seamlessly.")

st.sidebar.title("🧭 Navigation")
app_choice = st.sidebar.radio(
    "Select a module:",
    [
        "📄 Document Detection",
        "📋 KYC Classification & Quality",
        "🧠 Document Summarizer"
    ]
)

st.sidebar.divider()

# =========================
# Helper Functions (Common)
# =========================
def create_zip_from_folder(folder_path, zip_name):
    zip_path = Path(folder_path).parent / zip_name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zipf.write(file_path, arcname)
    return zip_path


# =========================
# 1️⃣ Document Detection Module
# =========================
def document_detection_ui():
    st.title("📄 Document Detection System")
    st.markdown("**Extract Stamps, Signatures, Photos, and OCR data from PDFs**")

    if 'detector' not in st.session_state:
        st.session_state.detector = None
        st.session_state.processed = False
        st.session_state.results_dir = None
        st.session_state.last_upload_count = 0

    def initialize_detector():
        if st.session_state.detector is None:
            with st.spinner("Loading detection models... This may take a minute."):
                st.session_state.detector = ObjectDetection()
            st.success("✓ Models loaded successfully!")

    def clear_previous_results(temp_dir):
        temp_path = Path(temp_dir)
        if temp_path.exists():
            shutil.rmtree(temp_path)
            st.info("🗑️ Cleared previous results")
        temp_path.mkdir(exist_ok=True)
        st.session_state.processed = False
        st.session_state.results_dir = None

    def display_results(output_dir, detection_type):
        output_path = Path(output_dir)
        if not output_path.exists():
            st.warning(f"No {detection_type} results found.")
            return

        image_files = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))
        json_files = list(output_path.glob("*.json"))

        st.subheader(f"📊 {detection_type.upper()} Results")

        if json_files:
            with st.expander(f"📄 View {detection_type.upper()} Metadata"):
                for json_file in json_files:
                    st.markdown(f"**{json_file.name}**")
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        st.json(data)

        if image_files:
            st.markdown(f"**Detected {detection_type.upper()}s:**")
            vis_images = [img for img in image_files if 'vis' in img.name]
            crop_images = [img for img in image_files if 'vis' not in img.name]

            if vis_images:
                st.markdown("**📸 Visualizations (with bounding boxes):**")
                cols = st.columns(min(3, len(vis_images)))
                for idx, img_path in enumerate(vis_images):
                    with cols[idx % 3]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)

            if crop_images:
                st.markdown(f"**✂️ Extracted {detection_type.upper()}s:**")
                cols = st.columns(min(4, len(crop_images)))
                for idx, img_path in enumerate(crop_images):
                    with cols[idx % 4]:
                        st.image(str(img_path), caption=img_path.name, use_container_width=True)

        if image_files or json_files:
            zip_path = create_zip_from_folder(output_path, f"{detection_type}_results.zip")
            with open(zip_path, "rb") as f:
                st.download_button(
                    label=f"⬇️ Download {detection_type.upper()} Results (ZIP)",
                    data=f,
                    file_name=f"{detection_type}_results.zip",
                    mime="application/zip"
                )

    # Sidebar options
    with st.sidebar:
        st.header("⚙️ Detection Settings")
        detect_stamp = st.checkbox("🔖 Stamps", value=True)
        detect_signature = st.checkbox("✍️ Signatures", value=True)
        detect_photo = st.checkbox("📷 Photos", value=True)
        detect_ocr = st.checkbox("📝 OCR (Text Extraction)", value=False)

    uploaded_files = st.file_uploader("📤 Upload PDF Files", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        current_upload_count = len(uploaded_files)
        temp_dir = Path("temp_uploads")
        if current_upload_count != st.session_state.last_upload_count:
            clear_previous_results(temp_dir)
            st.session_state.last_upload_count = current_upload_count

        st.success(f"✓ {len(uploaded_files)} file(s) uploaded successfully!")
        temp_dir.mkdir(exist_ok=True)

        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
            st.info(f"📁 Saved: {uploaded_file.name}")

        if st.button("🚀 Start Processing", type="primary"):
            initialize_detector()
            detection_types = []
            if detect_stamp: detection_types.append("stamp")
            if detect_signature: detection_types.append("signature")
            if detect_photo: detection_types.append("photo")
            if detect_ocr: detection_types.append("ocr")

            if not detection_types:
                st.error("⚠️ Please select at least one detection type!")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()

                for idx, file_path in enumerate(saved_files):
                    status_text.text(f"Processing {file_path.name}...")
                    for det in detection_types:
                        output_dir = temp_dir / f"{det}_outputs"
                        try:
                            if det == "stamp":
                                st.session_state.detector.detect_stamps(str(file_path), str(output_dir))
                            elif det == "signature":
                                st.session_state.detector.detect_signatures_tiled(str(file_path), str(output_dir))
                            elif det == "photo":
                                st.session_state.detector.detect_photos(str(file_path), str(output_dir))
                            elif det == "ocr":
                                st.session_state.detector.detect_ocr(str(file_path), str(output_dir))
                        except Exception as e:
                            st.error(f"✗ Error: {e}")
                    progress_bar.progress((idx + 1) / len(saved_files))

                status_text.text("✓ Processing completed!")
                st.session_state.processed = True
                st.session_state.results_dir = temp_dir

    if st.session_state.processed:
        st.header("📊 Detection Results")
        tabs = []
        if detect_stamp: tabs.append("🔖 Stamps")
        if detect_signature: tabs.append("✍️ Signatures")
        if detect_photo: tabs.append("📷 Photos")
        if detect_ocr: tabs.append("📝 OCR")

        tab_objs = st.tabs(tabs)
        idx = 0
        if detect_stamp:
            with tab_objs[idx]: display_results(st.session_state.results_dir / "stamp_outputs", "stamp"); idx += 1
        if detect_signature:
            with tab_objs[idx]: display_results(st.session_state.results_dir / "signature_outputs", "signature"); idx += 1
        if detect_photo:
            with tab_objs[idx]: display_results(st.session_state.results_dir / "photo_outputs", "photo"); idx += 1
        if detect_ocr:
            with tab_objs[idx]: display_results(st.session_state.results_dir / "ocr_outputs", "ocr")

# =========================
# 2️⃣ KYC Classification & Quality Module
# =========================
def kyc_quality_ui():
    from kyc_classifier import classification_results, quality_results  # reuse functions if modularized

    st.title("📋 KYC Document Analyzer")
    st.markdown("**Classify document types and assess quality for KYC processing**")

    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
        st.session_state.processed = False

    def initialize_classifier():
        if st.session_state.classifier is None:
            with st.spinner("Loading KYC Classification and Quality Check models..."):
                st.session_state.classifier = KYCClassificationQuality()
            st.success("✓ Models loaded successfully!")

    uploaded_files = st.file_uploader("📤 Upload PDF Documents", type=['pdf'], accept_multiple_files=True)
    if uploaded_files:
        temp_dir = Path("temp_kyc_uploads")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
            st.info(f"📁 Saved: {uploaded_file.name}")

        if st.button("🚀 Analyze Documents", type="primary"):
            initialize_classifier()
            progress_bar = st.progress(0)
            status_text = st.empty()
            classification_dir = temp_dir / "kyc_classify"
            quality_dir = temp_dir / "doc_quality"

            for idx, file_path in enumerate(saved_files):
                status_text.text(f"Analyzing {file_path.name}...")
                try:
                    st.session_state.classifier.process_pdf(str(file_path), str(classification_dir), str(quality_dir))
                    st.success(f"✓ Completed: {file_path.name}")
                except Exception as e:
                    st.error(f"✗ Error: {e}")
                progress_bar.progress((idx + 1) / len(saved_files))

            status_text.text("✓ Analysis completed!")
            st.session_state.processed = True
            st.session_state.classification_results = classification_dir
            st.session_state.quality_results = quality_dir

    if st.session_state.processed:
        tab1, tab2 = st.tabs(["📊 Classification", "🔍 Quality"])
        with tab1:
            classification_results(st.session_state.classification_results)
        with tab2:
            quality_results(st.session_state.quality_results)



def kyc_quality_ui():
    """
    KYC Classification & Quality UI.
    Uses KYCClassificationQuality from kyc_classifier.py and displays results saved to JSON.
    """

    st.title("📋 KYC Document Analyzer")
    st.markdown("**Classify document types and assess quality for KYC processing**")

    # session-state setup
    if 'classifier' not in st.session_state:
        st.session_state.classifier = None
        st.session_state.processed_kyc = False
        st.session_state.classification_results = None
        st.session_state.quality_results = None
        st.session_state.last_kyc_upload_count = 0

    def initialize_classifier():
        if st.session_state.classifier is None:
            with st.spinner("Loading KYC Classification and Quality Check models..."):
                st.session_state.classifier = KYCClassificationQuality()
            st.success("✓ Models loaded successfully!")

    def clear_previous_kyc_results():
        temp_dir = Path("temp_kyc_uploads")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            st.info("🗑️ Cleared previous results")
        temp_dir.mkdir(exist_ok=True)
        st.session_state.processed_kyc = False
        st.session_state.classification_results = None
        st.session_state.quality_results = None

    # Reuse display functions from your original KYC UI (with utf-8 file open)
    def display_classification_results(classification_dir):
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
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    st.error(f"Failed to read {json_file.name}: {e}")
                    continue

                st.markdown(f"**PDF:** `{data.get('pdf','-')}`")
                st.markdown(f"**Total Pages:** {data.get('total_pages','-')}")

                for page_data in data.get('pages', []):
                    page_num = page_data.get('page', '-')
                    classification = page_data.get('classification', {})

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Page", page_num)
                    with col2:
                        st.metric(
                            "Document Type",
                            classification.get('kyc_category', classification.get('base_type', '-')),
                            help=f"Base Type: {classification.get('base_type','-')}"
                        )
                    with col3:
                        conf = classification.get('confidence', 0.0)
                        try:
                            confidence_pct = f"{conf*100:.1f}%"
                        except Exception:
                            confidence_pct = str(conf)
                        st.metric("Confidence", confidence_pct)

                    if classification.get('is_valid_kyc'):
                        st.success("✅ Valid KYC Document")
                    else:
                        st.warning("⚠️ Not a KYC Document")
                    st.divider()

    def display_quality_results(quality_dir):
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
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    st.error(f"Failed to read {json_file.name}: {e}")
                    continue

                st.markdown(f"**PDF:** `{data.get('pdf','-')}`")
                st.markdown(f"**Total Pages:** {data.get('total_pages','-')}")

                for page_data in data.get('pages', []):
                    page_num = page_data.get('page', '-')
                    quality = page_data.get('quality', {})

                    st.markdown(f"### Page {page_num}")

                    overall_score = quality.get('overall_score', 0.0)
                    quality_label = quality.get('quality_label', '-')

                    if overall_score >= 0.8:
                        score_color = "🟢"
                    elif overall_score >= 0.6:
                        score_color = "🟡"
                    else:
                        score_color = "🔴"

                    st.markdown(f"### {score_color} Overall Quality: **{quality_label}** ({overall_score:.2f})")
                    st.info(quality.get('recommendation', ''))

                    # Detailed metrics safely accessed
                    st.markdown("#### Quality Metrics:")
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        res = quality.get('resolution', {})
                        st.metric(
                            "Resolution",
                            res.get('status','-'),
                            delta=f"{res.get('megapixels','-')} MP",
                            help=f"{res.get('width','-')}x{res.get('height','-')} pixels"
                        )
                        bright = quality.get('brightness', {})
                        st.metric("Brightness", bright.get('status','-'), delta=f"{bright.get('mean_value','-')}/255")

                    with col2:
                        contrast = quality.get('contrast', {})
                        st.metric("Contrast", contrast.get('status','-'), delta=f"σ={contrast.get('std_deviation','-')}")
                        sharp = quality.get('sharpness', {})
                        st.metric("Sharpness", sharp.get('status','-'), delta=f"Var={sharp.get('laplacian_variance','-')}")

                    with col3:
                        noise = quality.get('noise', {})
                        st.metric("Noise Level", noise.get('status','-'), delta=f"{noise.get('noise_level','-')}")
                        skew = quality.get('skew', {})
                        st.metric("Alignment", skew.get('status','-'), delta=f"{skew.get('angle','-')}°")

                    st.markdown("#### Score Breakdown:")
                    scores = {
                        "Resolution": quality.get('resolution', {}).get('score', 0.0),
                        "Brightness": quality.get('brightness', {}).get('score', 0.0),
                        "Contrast": quality.get('contrast', {}).get('score', 0.0),
                        "Sharpness": quality.get('sharpness', {}).get('score', 0.0),
                        "Noise": quality.get('noise', {}).get('score', 0.0),
                        "Skew": quality.get('skew', {}).get('score', 0.0)
                    }

                    # Use st.progress for each metric (note: progress expects int 0-100 or float 0.0-1.0)
                    for metric, score in scores.items():
                        try:
                            st.progress(score, text=f"{metric}: {score:.2f}")
                        except Exception:
                            # fallback: show as text
                            st.markdown(f"- {metric}: {score:.2f}")

                    st.divider()

    # Sidebar informative text
    with st.sidebar:
        st.header("⚙️ KYC Settings")
        st.info("Both classification and quality check are performed automatically.")

    # File uploader & processing
    uploaded_files = st.file_uploader(
        "📤 Upload PDF Documents for KYC Analysis",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload one or more PDF documents for classification and quality assessment"
    )

    if uploaded_files:
        current_upload_count = len(uploaded_files)
        if current_upload_count != st.session_state.last_kyc_upload_count:
            clear_previous_kyc_results()
            st.session_state.last_kyc_upload_count = current_upload_count

        st.success(f"✓ {len(uploaded_files)} file(s) uploaded successfully!")

        temp_dir = Path("temp_kyc_uploads")
        temp_dir.mkdir(exist_ok=True)

        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
            st.info(f"📁 Saved: {uploaded_file.name}")

        col1, col2 = st.columns([3, 1])
        with col1:
            process_button = st.button("🚀 Analyze Documents", type="primary")
        with col2:
            if st.button("🗑️ Clear Results"):
                clear_previous_kyc_results() 
                st.session_state.processed = False
                st.session_state.classification_results = None
                st.session_state.quality_results = None
                st.session_state.last_upload_count = 0
                st.success("✅ Cleared all previous results!")
                st.rerun()

        if process_button:
            initialize_classifier()
            progress_bar = st.progress(0)
            status_text = st.empty()

            classification_dir = temp_dir / "kyc_classify"
            quality_dir = temp_dir / "doc_quality"
            classification_dir.mkdir(parents=True, exist_ok=True)
            quality_dir.mkdir(parents=True, exist_ok=True)

            for idx, file_path in enumerate(saved_files):
                status_text.text(f"Analyzing {file_path.name}...")
                try:
                    # process_pdf returns (classification_results, quality_results) but also writes JSON files
                    classification_results, quality_results = st.session_state.classifier.process_pdf(
                        str(file_path), str(classification_dir), str(quality_dir)
                    )
                    st.success(f"✓ Completed: {file_path.name}")
                except Exception as e:
                    st.error(f"✗ Error processing {file_path.name}: {e}")
                progress_bar.progress((idx + 1) / len(saved_files))

            status_text.text("✓ All files analyzed!")
            st.session_state.processed_kyc = True
            st.session_state.classification_results = classification_dir
            st.session_state.quality_results = quality_dir

    # Display results if present
    if st.session_state.processed_kyc:
        st.divider()
        st.header("📊 Analysis Results")
        tab1, tab2 = st.tabs(["📊 Classification", "🔍 Quality Assessment"])
        with tab1:
            if st.session_state.classification_results:
                display_classification_results(st.session_state.classification_results)
        with tab2:
            if st.session_state.quality_results:
                display_quality_results(st.session_state.quality_results)

        # Provide downloads
        st.divider()
        st.subheader("📥 Download Results")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.classification_results and Path(st.session_state.classification_results).exists():
                classification_zip = Path("temp_kyc_uploads") / "classification_results.zip"
                with zipfile.ZipFile(classification_zip, 'w') as zipf:
                    for file in Path(st.session_state.classification_results).glob("*.json"):
                        zipf.write(file, file.name)
                with open(classification_zip, 'rb') as f:
                    st.download_button("📊 Download Classification Results", data=f, file_name="classification_results.zip", mime="application/zip")
        with col2:
            if st.session_state.quality_results and Path(st.session_state.quality_results).exists():
                quality_zip = Path("temp_kyc_uploads") / "quality_results.zip"
                with zipfile.ZipFile(quality_zip, 'w') as zipf:
                    for file in Path(st.session_state.quality_results).glob("*.json"):
                        zipf.write(file, file.name)
                with open(quality_zip, 'rb') as f:
                    st.download_button("🔍 Download Quality Results", data=f, file_name="quality_results.zip", mime="application/zip")




# =========================
# 3️⃣ Placeholder: Summarizer
# =========================
def summarizer_ui():
    st.title("🧠 Document Summarizer")
    # st.info("Coming soon: Upload documents and generate intelligent summaries automatically.")
    st.markdown("Upload financial or analytical PDFs to generate AI- powered summaries")
    st.markdown("Summaries are automatically saved under 'data/doc_summary_outputs/.'")
    
    if 'summarizer' not in st.session_state:
        st.session_state.summarizer = None
        st.session_state.processed_sum = False
        st.session_state.summary_outputs = None

    # Initialize summarizer
    def initialize_summarizer():
        if st.session_state.summarizer is None:
            with st.spinner("Initializing summarization models..."):
                st.session_state.summarizer = Doc_Summarizer()
            st.success("✅ Summarizer initialized successfully!")

    uploaded_files = st.file_uploader(
        "📤 Upload PDF(s) for Summarization",
        type=['pdf'],
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"✓ {len(uploaded_files)} file(s) uploaded successfully!")
        temp_dir = Path("temp_summarizer_uploads")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(exist_ok=True)

        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
            st.info(f"📁 Saved: {uploaded_file.name}")

        if st.button("🚀 Generate Summaries", type="primary"):
            initialize_summarizer()
            st.info("⏳ Starting summarization pipeline...")
            progress_bar = st.progress(0)
            status_text = st.empty()

            topic_json_path = "topic_config.json"
            # Save the topic JSON inline for this run
            topic_json = {
                "Company Overview": {
                    "retrieval_prompts": ["About the company", "Business overview"]
                },
                "Financial Highlights": {
                    "retrieval_prompts": ["Financial highlights", "Performance overview"]
                }
            }

            # Optional — save topic json so user can modify later
            with open(topic_json_path, "w", encoding="utf-8") as f:
                json.dump(topic_json, f, indent=2)

            results = []
            for idx, file_path in enumerate(saved_files):
                try:
                    # update doc_sum_pipeline to take dynamic pdf_path
                    st.session_state.summarizer.pdf_path = str(file_path)
                    summary = st.session_state.summarizer.summarize_document_workflow(topic_json)
                    results.append((file_path.name, summary))
                    st.success(f"✓ Summary generated: {file_path.name}")
                except Exception as e:
                    st.error(f"✗ Error summarizing {file_path.name}: {e}")
                progress_bar.progress((idx + 1) / len(saved_files))

            st.session_state.processed_sum = True
            st.session_state.summary_outputs = Path("data/doc_summary_outputs")

    if st.session_state.processed_sum and st.session_state.summary_outputs:
        st.divider()
        st.header("📊 Generated Summaries")
        output_path = Path(st.session_state.summary_outputs)

        for txt_file in output_path.glob("*.txt"):
            with st.expander(f"📄 {txt_file.name}", expanded=False):
                with open(txt_file, "r", encoding="utf-8") as f:
                    st.text(f.read())

        # Download all summaries as ZIP
        zip_path = create_zip_from_folder(output_path, "document_summaries.zip")
        with open(zip_path, "rb") as f:
            st.download_button(
                label="⬇️ Download All Summaries (ZIP)",
                data=f,
                file_name="document_summaries.zip",
                mime="application/zip"
            )

# =========================
# Page Routing
# =========================
if "Detection" in app_choice:
    document_detection_ui()
elif "KYC" in app_choice:
    kyc_quality_ui()
else:
    summarizer_ui()

# =========================
# Footer
# =========================
st.divider()
st.markdown(
    "<div style='text-align:center;color:gray;'>📂 Unified Document CoPilot • Detection • Classification • Quality • Summarization</div>",
    unsafe_allow_html=True
)
