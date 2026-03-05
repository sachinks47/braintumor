import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB2
from PIL import Image
import base64
from io import BytesIO
import datetime
import tempfile
import os
import time

# Try to load FPDF globally
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

# ---------------------------------------------------
# PAGE CONFIG (WIDE)
# ---------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded" 
)

IMG_SIZE = 260
CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
MODEL_ACCURACY = "87.12%"

# ---------------------------------------------------
# THEME STATE MANAGEMENT
# ---------------------------------------------------
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

theme = st.session_state.theme

# Define Dynamic Colors Based on Theme
if theme == 'dark':
    colors = {
        "bg": "linear-gradient(135deg, #0f2027, #203a43, #2c5364)",
        "text": "white",
        "card_bg": "rgba(255, 255, 255, 0.05)",
        "card_border": "rgba(255, 255, 255, 0.1)",
        "shadow": "0 4px 30px rgba(0, 0, 0, 0.1)",
        "hero_grad": "-webkit-linear-gradient(#ffffff, #a1c4fd)",
        "sub_text": "#b0bec5",
        "stat_bg": "rgba(0, 0, 0, 0.2)",
        "stat_border": "rgba(255, 255, 255, 0.05)",
        "upload_border": "rgba(255, 255, 255, 0.3)",
        "upload_hover": "rgba(255, 255, 255, 0.08)",
        "prog_bg": "rgba(255,255,255,0.1)",
        "divider": "rgba(255,255,255,0.1)",
        "info_card": "rgba(255, 255, 255, 0.03)",
        "btn_bg": "linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%)" # Red gradient
    }
else:
    colors = {
        "bg": "linear-gradient(135deg, #eef2f3, #dce6ed)",
        "text": "#1e293b",
        "card_bg": "rgba(255, 255, 255, 0.65)",
        "card_border": "rgba(255, 255, 255, 0.9)",
        "shadow": "0 8px 32px rgba(31, 38, 135, 0.05)",
        "hero_grad": "-webkit-linear-gradient(#0f172a, #2563eb)",
        "sub_text": "#64748b",
        "stat_bg": "rgba(255, 255, 255, 0.9)",
        "stat_border": "rgba(226, 232, 240, 1)",
        "upload_border": "rgba(100, 116, 139, 0.3)",
        "upload_hover": "rgba(255, 255, 255, 1)",
        "prog_bg": "rgba(0,0,0,0.06)",
        "divider": "rgba(0,0,0,0.08)",
        "info_card": "rgba(255, 255, 255, 0.5)",
        "btn_bg": "linear-gradient(90deg, #dc2626 0%, #ef4444 100%)" # Red gradient
    }

# ---------------------------------------------------
# PREMIUM GLASSMORPHISM & UI STYLING
# ---------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background: {colors['bg']};
    color: {colors['text']};
    transition: background 0.5s ease;
}}

/* Clean up Streamlit UI */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}

/* Fix unclickable theme button and completely remove the top black box */
header[data-testid="stHeader"], .stAppHeader {{
    background: transparent !important;
    background-color: transparent !important;
    pointer-events: none !important; 
}}
header[data-testid="stHeader"] *, .stAppHeader * {{
    pointer-events: auto !important;
}}

.block-container {{
    padding-top: 0rem;
    padding-left: 3rem;
    padding-right: 3rem;
}}

/* Typography */
.hero-title {{
    text-align: center;
    font-size: 3.5rem;
    font-weight: 700;
    background: {colors['hero_grad']};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 80px; /* Increased from -10px to add space above the heading */
    margin-bottom: 5px;
    letter-spacing: -1px;
}}

.hero-subtitle {{
    text-align: center;
    font-size: 1.2rem;
    color: {colors['sub_text']};
    margin-bottom: 40px;
    font-weight: 300;
}}

/* Glassmorphism Cards */
.glass-card {{
    background: {colors['card_bg']};
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-radius: 16px;
    border: 1px solid {colors['card_border']};
    padding: 30px;
    box-shadow: {colors['shadow']};
    height: 100%;
}}

.content-card {{
    background: {colors['info_card']};
    border-radius: 12px;
    padding: 20px;
    border: 1px solid {colors['divider']};
    margin-bottom: 20px;
    text-align: left;
}}

/* File Uploader Target Override */
div[data-testid="stFileUploader"] > div > div > div,
div[data-testid="stFileUploader"] section {{
    background-color: {colors['card_bg']} !important;
    border-radius: 15px !important;
    border: 1.5px dashed {colors['upload_border']} !important;
    transition: all 0.3s ease !important;
}}
div[data-testid="stFileUploader"] > div > div > div:hover,
div[data-testid="stFileUploader"] section:hover {{
    border-color: {colors['text']} !important;
    background-color: {colors['upload_hover']} !important;
}}
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p {{
    color: {colors['text']} !important;
}}
div[data-testid="stFileUploader"] button {{
    color: {colors['text']} !important;
    border-color: {colors['card_border']} !important;
    background-color: transparent !important;
}}
div[data-testid="stFileUploader"] svg {{
    fill: {colors['text']} !important;
    color: {colors['text']} !important;
}}

/* Download Button Styling (Solid Red Gradient) */
div[data-testid="stDownloadButton"] button {{
    background: {colors['btn_bg']} !important;
    color: white !important;
    border: none !important;
    padding: 20px !important;
    border-radius: 12px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}}
div[data-testid="stDownloadButton"] button:hover {{
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0,0,0,0.2) !important;
}}

/* Form Submit Button Styling (Generate Report - Glassmorphic Red) */
div[data-testid="stForm"] button {{
    background: rgba(239, 68, 68, 0.15) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(239, 68, 68, 0.4) !important;
    color: {colors['text']} !important;
    padding: 15px !important;
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}}
div[data-testid="stForm"] button:hover {{
    background: rgba(239, 68, 68, 0.3) !important;
    transform: translateY(-2px) !important;
    border: 1px solid rgba(239, 68, 68, 0.6) !important;
    box-shadow: 0 8px 25px rgba(239, 68, 68, 0.25) !important;
}}

/* Mobile Optimization via Media Queries */
@media (max-width: 768px) {{
    .hero-title {{
        font-size: 2.2rem !important;
    }}
    .hero-subtitle {{
        font-size: 1rem !important;
        margin-bottom: 25px;
    }}
    .block-container {{
        padding-left: 1rem;
        padding-right: 1rem;
    }}
    .stat-container {{
        flex-direction: column;
    }}
}}

/* =========================================
   CUSTOM CSS LOADER
   ========================================= */
.loader-container {{
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 50px;
    margin-bottom: 50px;
}}
.loader {{
  display: flex;
  gap: 8px;
  justify-content: center;
  align-items: center;
  height: 60px;
}}
.loader-square {{
  width: 22px;
  height: 22px;
  background-color: rgb(0, 247, 255);
  border-radius: 4px;
  box-shadow: 0 0 12px rgba(4, 136, 252, 0.8);
  animation: scaleBounce 1.2s infinite ease-in-out;
  position: relative;
}}
.loader-square::after {{
  content: "";
  position: absolute;
  width: 100%;
  height: 100%;
  background-color: rgba(0, 247, 255, 0.5);
  border-radius: 50%;
  opacity: 0;
  transform: scale(1);
  animation: splash 1.2s infinite ease-in-out;
}}
.loader-square:nth-child(1) {{
  animation-delay: -0.4s;
}}
.loader-square:nth-child(2) {{
  animation-delay: -0.2s;
}}
.loader-square:nth-child(3) {{
  animation-delay: 0s;
}}
@keyframes scaleBounce {{
  0%,
  80%,
  100% {{
    transform: scale(0.5);
    opacity: 0.6;
  }}
  40% {{
    transform: scale(1.2);
    opacity: 1;
  }}
}}
@keyframes splash {{
  0% {{
    opacity: 0.6;
    transform: scale(1);
  }}
  50% {{
    opacity: 0;
    transform: scale(2);
  }}
  100% {{
    opacity: 0;
    transform: scale(2.5);
  }}
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPER FUNCTIONS (Base64 & PDF Generation)
# ---------------------------------------------------
def get_image_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def generate_pdf_report(patient_name, date_str, predicted_label, confidence, prob_dict, img_pil):
    """Generates a professional PDF report using fpdf."""
    if not FPDF_AVAILABLE:
        return None  # Gracefully fail and show warning in UI
        
    pdf = FPDF()
    pdf.add_page()
    
    # Add Title
    pdf.set_font("Arial", 'B', 22)
    pdf.cell(200, 10, txt="Brain Tumor Analysis Report", ln=True, align='C')
    pdf.ln(5)
    
    # Add Patient Info
    pdf.set_font("Arial", '', 12)
    display_name = patient_name if patient_name.strip() else "Not Provided"
    pdf.cell(200, 8, txt=f"Patient Name: {display_name}", ln=True, align='L')
    pdf.cell(200, 8, txt=f"Date of Analysis: {date_str}", ln=True, align='L')
    pdf.ln(10)
    
    # Save Image to temp file and attach to PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        img_pil.save(tmp_file.name)
        pdf.image(tmp_file.name, x=65, y=55, w=80)
    
    pdf.ln(90) # Skip past the image height
    
    # Add Results
    pdf.set_font("Arial", 'B', 16)
    res = "Negative" if predicted_label == "No Tumor" else "Positive"
    pdf.cell(200, 10, txt=f"Result: {res}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Type: {predicted_label}", ln=True, align='C')
    pdf.ln(5)
    
    # Add Metrics
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 8, txt=f"Confidence Score: {confidence:.2f}%", ln=True, align='C')
    pdf.ln(10)
    
    # Add Probability Distribution
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="Probability Distribution:", ln=True, align='L')
    pdf.set_font("Arial", '', 12)
    for label, prob in prob_dict.items():
        pdf.cell(200, 8, txt=f"  - {label}: {prob:.2f}%", ln=True, align='L')
        
    pdf.ln(15)
    pdf.set_font("Arial", 'I', 10)
    pdf.multi_cell(0, 6, txt="Disclaimer: This report is generated by an AI model for educational and research purposes only. It is not a clinical diagnosis. Please consult a qualified radiologist or physician.")
    
    # Extract bytes directly using temp file to avoid fpdf versioning conflicts
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
        pdf.output(tmp_pdf.name)
    with open(tmp_pdf.name, "rb") as f:
        pdf_bytes = f.read()
        
    # Cleanup temp files
    os.remove(tmp_file.name)
    os.remove(tmp_pdf.name)
    
    return pdf_bytes

# ---------------------------------------------------
# SIDEBAR (PROFESSIONAL DOCUMENTATION)
# ---------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3003/3003058.png", width=60) # Placeholder medical icon
    
    st.markdown("## 🎨 Appearance")
    btn_label = "☀️ Switch to Light Mode" if st.session_state.theme == 'dark' else "🌙 Switch to Dark Mode"
    if st.button(btn_label, use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
            
    st.markdown("---")
    
    st.markdown("## App Settings & Info")
    st.markdown("---")
    
    st.markdown("### 📋 About this App")
    st.markdown("This application utilizes deep learning to analyze structural magnetic resonance imaging (MRI) of the brain and classify the presence and type of tumors.")
    
    st.markdown("### ⚙️ Model Architecture")
    st.markdown("- **Base Model:** EfficientNetB2")
    st.markdown("- **Pre-processing:** CLAHE Enhancement, Gaussian Blur, Contour Cropping")
    st.markdown(f"- **Validation Accuracy:** {MODEL_ACCURACY}")
    
    st.markdown("---")
    st.warning("⚠️ **Medical Disclaimer**\n\nThis tool is for educational and research purposes only. It is **not** a diagnostic tool. Always consult a qualified radiologist or medical professional for clinical diagnosis.")

# ---------------------------------------------------
# HERO SECTION
# ---------------------------------------------------
st.markdown("<div class='hero-title'>Brain Tumor Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>AI-Powered Clinical Classification System</div>", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD MODEL (Cached)
# ---------------------------------------------------
@st.cache_resource
def load_model():
    base_model = EfficientNetB2(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(4, activation="softmax")(x)

    model = models.Model(inputs, outputs)

    try:
        model.load_weights("model.weights.h5")
    except Exception as e:
        pass # Silently pass in UI, add logic here if needed

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = load_model()

# ---------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------
def advanced_preprocess_cv2(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return image_np
        
    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    x, y, w, h = cv2.boundingRect(c)
    cropped = masked[y:y+h, x:x+w]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(cropped)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

# ---------------------------------------------------
# UPLOADER SECTION (CENTERED)
# ---------------------------------------------------
col_spacer1, col_uploader, col_spacer2 = st.columns([1, 2, 1])

with col_uploader:
    st.markdown(f"<div style='text-align: center; margin-bottom: 10px; color: {colors['text']}; font-weight: 600; font-size: 16px;'>Upload MRI Scan (Axial View Preferred)</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------
# MAIN CONTENT AREA (DYNAMIC)
# ---------------------------------------------------
if not uploaded_file:
    # --- EMPTY STATE (PROFESSIONAL LANDING CONTENT) ---
    col_info1, col_info2 = st.columns(2, gap="large")
    
    with col_info1:
        st.markdown(f"""
        <div class='content-card'>
            <h3 style='margin-top:0; color:{colors["text"]};'>🔬 Supported Classifications</h3>
            <p style='color:{colors["sub_text"]}; font-size: 14px;'>Our neural network is trained to identify and categorize three major types of brain tumors, alongside healthy scans.</p>
            <ul style='color:{colors["text"]}; font-size: 15px; line-height: 1.8;'>
                <li><b>Glioma:</b> Tumors that occur in the brain and spinal cord.</li>
                <li><b>Meningioma:</b> Tumors arising from the meninges (membranes surrounding brain/spinal cord).</li>
                <li><b>Pituitary:</b> Tumors developing in the pituitary gland.</li>
                <li><b>No Tumor:</b> Healthy brain MRI classification.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown(f"""
        <div class='content-card'>
            <h3 style='margin-top:0; color:{colors["text"]};'>⚙️ How it works</h3>
            <ol style='color:{colors["text"]}; font-size: 15px; line-height: 1.8;'>
                <li><b>Upload:</b> Provide a clear, top-down (axial) MRI scan in JPG or PNG format.</li>
                <li><b>Pre-Processing:</b> The system automatically crops excess black space and applies CLAHE enhancement to isolate brain tissue.</li>
                <li><b>Inference:</b> The EfficientNetB2 deep learning model analyzes structural anomalies.</li>
                <li><b>Result:</b> View confidence scores and probability distributions instantly.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

else:
    # --- ANIMATED LOADER ---
    # Temporarily display the loader while the image is being processed
    loader_placeholder = st.empty()
    loader_placeholder.markdown("""
        <div class="loader-container">
            <div class="loader">
                <div class="loader-square"></div>
                <div class="loader-square"></div>
                <div class="loader-square"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    time.sleep(1.5) # Slight delay to let the animation show smoothly

    # --- ANALYSIS STATE (RESULTS) ---
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_b64 = get_image_base64(image)

    # Preprocessing & Inference
    img = advanced_preprocess_cv2(img_array)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32)

    mean = np.mean(img)
    std = np.std(img) + 1e-7
    img = (img - mean) / std
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    predicted_index = np.argmax(prediction[0])
    predicted_label = CLASSES[predicted_index]
    confidence = float(np.max(prediction[0])) * 100

    # Hide the loader once inference completes
    loader_placeholder.empty()

    col1, col2 = st.columns([1, 1.5], gap="large")

    # LEFT PANEL (IMAGE)
    with col1:
        st.markdown(f"""
        <div class='glass-card' style='display: flex; flex-direction: column; align-items: center; justify-content: center;'>
            <h4 style='margin-top:0; font-weight:600; color:{colors["text"]}; font-size:18px; margin-bottom: 20px; align-self: flex-start;'>Uploaded MRI</h4>
            <img src="data:image/png;base64,{img_b64}" style="width:100%; max-width: 350px; border-radius:12px; box-shadow: 0 4px 15px rgba(0,0,0,0.2);" alt="MRI Scan">
        </div>
        """, unsafe_allow_html=True)

    # RIGHT PANEL (RESULTS)
    with col2:
        if predicted_label == "No Tumor":
            badge_html = f"""
<div style='background-color: rgba(0, 230, 118, 0.15); color: #00b35c; padding: 25px; border-radius: 12px; border: 1px solid rgba(0, 230, 118, 0.3); text-align: center; margin-bottom: 20px;'>
<div style='font-size: 28px; margin-bottom: 12px;'><b>Result :</b> Negative</div>
<div style='font-size: 22px;'><b>Type :</b> None Detected</div>
</div>
"""
        else:
            badge_html = f"""
<div style='background-color: rgba(255, 82, 82, 0.15); color: #ff5252; padding: 25px; border-radius: 12px; border: 1px solid rgba(255, 82, 82, 0.3); text-align: center; margin-bottom: 20px;'>
<div style='font-size: 28px; margin-bottom: 12px;'><b>Result :</b> Positive</div>
<div style='font-size: 22px;'><b>Type :</b> {predicted_label}</div>
</div>
"""

        bars_html = ""
        prob_dict = {} # Collected for the PDF report
        
        for i, label in enumerate(CLASSES):
            prob = float(prediction[0][i]) * 100
            prob_dict[label] = prob
            
            # Streamlit UI Bars
            bars_html += f"""
<div style="margin-bottom: 15px;">
<div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 6px; color: {colors['sub_text']};">
<span style="font-weight: 500;">{label}</span>
<span style="font-weight: 600;">{prob:.2f}%</span>
</div>
<div style="background: {colors['prog_bg']}; border-radius: 10px; width: 100%; height: 8px;">
<div style="background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); width: {prob}%; height: 100%; border-radius: 10px; transition: width 0.5s ease-in-out;"></div>
</div>
</div>
"""

        st.markdown(f"""
<div class='glass-card'>
<h4 style='margin-top:0; font-weight:600; color:{colors["text"]}; font-size:18px; margin-bottom: 20px;'>Analysis Results</h4>

{badge_html}

<div style="display: flex; gap: 15px; margin-bottom: 20px;" class="stat-container">
<div style="background: {colors['stat_bg']}; border-radius: 12px; padding: 20px; text-align: center; border: 1px solid {colors['stat_border']}; flex: 1;">
<div style="font-size: 28px; font-weight: 700; color: {colors['text']}; margin-bottom: 5px;">{confidence:.2f}%</div>
<div style="font-size: 12px; color: {colors['sub_text']}; text-transform: uppercase; letter-spacing: 1px; font-weight: 500;">Confidence Score</div>
</div>
</div>

<hr style="border-color: {colors['divider']}; margin: 25px 0;">

<h5 style='margin-top:0; font-weight:600; color:{colors["text"]}; font-size:16px; margin-bottom: 20px;'>Other Possibilities</h5>
{bars_html}

</div>
""", unsafe_allow_html=True)

    # ---------------------------------------------------
    # GENERATE DOWNLOADABLE PDF REPORT AREA
    # ---------------------------------------------------
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col_pdf_space1, col_pdf, col_pdf_space2 = st.columns([1, 2, 1])
    with col_pdf:
        st.markdown(f"<h3 style='text-align: center; color: {colors['text']};'>📄 Export Clinical Report</h3>", unsafe_allow_html=True)
        
        # Using a form explicitly fixes the 'lag' issue. The download button will only update when you hit Generate.
        with st.form(key="report_form"):
            patient_name = st.text_input("Patient Name (Optional):", placeholder="Enter Patient Name")
            submit_btn = st.form_submit_button("Generate Report File", use_container_width=True)
            
        if submit_btn:
            st.session_state.report_generated = True
            st.session_state.patient_name_final = patient_name

        if st.session_state.get("report_generated"):
            # Date String (DD-MM-YYYY)
            date_str = datetime.datetime.now().strftime("%d-%m-%Y")
            
            final_name = st.session_state.get("patient_name_final", "")
            
            # Format File Name
            safe_name = final_name.strip().replace(" ", "_") if final_name.strip() else "Unknown_Patient"
            file_name = f"{safe_name}_report_{date_str}.pdf"
            
            # Create PDF Bytes
            pdf_bytes = generate_pdf_report(final_name, date_str, predicted_label, confidence, prob_dict, image)
            
            if pdf_bytes:
                st.download_button(
                    label=f"📥 Download {file_name}",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                    use_container_width=True 
                )
            else:
                st.warning("⚠️ **Library Missing:** Please add `fpdf2` to your `requirements.txt` file (if deployed on HuggingFace) or run `pip install fpdf2` in your local terminal.")
