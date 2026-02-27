import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB2
from PIL import Image
import base64
from io import BytesIO
import time

# ---------------------------------------------------
# PAGE CONFIG (WIDE)
# ---------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

IMG_SIZE = 260
CLASSES = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
MODEL_ACCURACY = "98.5%" 

# ---------------------------------------------------
# THEME COLORS (DARK MODE)
# ---------------------------------------------------
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
    "divider": "rgba(255,255,255,0.1)"
}

# ---------------------------------------------------
# PREMIUM UI STYLING (FIXED)
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

.hero-title {{
    text-align: center;
    font-size: 3.5rem;
    font-weight: 700;
    background: {colors['hero_grad']};
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: -30px;
    margin-bottom: 5px;
    letter-spacing: -1px;
    line-height: 1.1;
}}

.hero-subtitle {{
    text-align: center;
    font-size: 1.2rem;
    color: {colors['sub_text']};
    margin-bottom: 40px;
    font-weight: 300;
    letter-spacing: 0.5px;
}}

.glass-card {{
    background: {colors['card_bg']};
    backdrop-filter: blur(16px);
    border-radius: 16px;
    border: 1px solid {colors['card_border']};
    padding: 30px;
    box-shadow: {colors['shadow']};
    transition: all 0.3s ease;
}}

.badge-positive {{
    background-color: rgba(255, 82, 82, 0.15);
    color: #ff5252;
    padding: 15px 20px;
    border-radius: 10px;
    border: 1px solid rgba(255, 82, 82, 0.3);
    font-weight: 500;
}}

.badge-negative {{
    background-color: rgba(0, 230, 118, 0.15);
    color: #00b35c;
    padding: 15px 20px;
    border-radius: 10px;
    border: 1px solid rgba(0, 230, 118, 0.3);
    font-weight: 500;
}}

.stat-container {{
    display: flex;
    gap: 15px;
    margin-top: 20px;
}}

.stat-box {{
    background: {colors['stat_bg']};
    border-radius: 12px;
    padding: 15px;
    flex: 1;
    text-align: center;
    border: 1px solid {colors['stat_border']};
}}

.stat-value {{
    font-size: 24px;
    font-weight: 700;
    color: {colors['text']};
}}

.stat-label {{
    font-size: 11px;
    color: {colors['sub_text']};
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}}

/* File Uploader Customization */
div[data-testid="stFileUploader"] section {{
    background-color: {colors['card_bg']} !important;
    border-radius: 15px !important;
    border: 1.5px dashed {colors['upload_border']} !important;
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------
def get_image_base64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ---------------------------------------------------
# LOAD MODEL
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
        st.warning(f"Note: Model weights 'model.weights.h5' not found. Inference will use random weights. Error: {e}")

    return model

model = load_model()

# ---------------------------------------------------
# IMAGE PROCESSING
# ---------------------------------------------------
def advanced_preprocess_cv2(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 3)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return image_np

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = gray[y:y+h, x:x+w]
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(cropped)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

# ---------------------------------------------------
# MAIN UI
# ---------------------------------------------------
st.markdown("<div class='hero-title'>Brain Tumor Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='hero-subtitle'>AI-Powered Clinical Classification System</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MRI Scan", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1, 1.5], gap="large")
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    img_b64 = get_image_base64(image)

    with col1:
        st.markdown(f"""
        <div class='glass-card'>
            <h4 style='margin-top:0; font-weight:600; color:{colors["text"]}; font-size:18px; margin-bottom: 20px;'>Uploaded MRI</h4>
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{img_b64}" style="width:100%; max-width: 350px; border-radius:12px;" alt="MRI Scan">
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # --- LOADING BAR LOGIC ---
        progress_bar = st.progress(0, text="Initializing AI Analysis...")
        time.sleep(0.3)

        # Preprocessing Step
        progress_bar.progress(30, text="Enhancing MRI scan quality & extracting contours...")
        processed = advanced_preprocess_cv2(img_array)
        processed = cv2.resize(processed, (IMG_SIZE, IMG_SIZE))
        processed = processed.astype(np.float32)
        time.sleep(0.3)

        # Normalization Step
        progress_bar.progress(60, text="Normalizing image tensors for deep learning...")
        processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-7)
        processed = np.expand_dims(processed, axis=0)
        time.sleep(0.3)

        # Inference Step
        progress_bar.progress(85, text="Running inference via EfficientNet architecture...")
        preds = model.predict(processed, verbose=0)
        idx = np.argmax(preds[0])
        label = CLASSES[idx]
        conf = float(np.max(preds[0])) * 100
        time.sleep(0.3)

        # Finalizing
        progress_bar.progress(100, text="Analysis complete! Generating clinical report...")
        time.sleep(0.4)
        progress_bar.empty() # Removes the loading bar once finished
        # ------------------------

        res_class = "badge-negative" if label == "No Tumor" else "badge-positive"
        res_text = "Negative" if label == "No Tumor" else "Positive"
        type_text = "None Detected" if label == "No Tumor" else label

        bars_html = ""
        for i, l in enumerate(CLASSES):
            prob = float(preds[0][i]) * 100
            # Removed indentation so Markdown doesn't treat this as a code block
            bars_html += f"""<div style="margin-bottom: 12px;">
<div style="display: flex; justify-content: space-between; font-size: 13px; color: {colors['sub_text']};">
<span>{l}</span><span>{prob:.1f}%</span>
</div>
<div style="background: {colors['prog_bg']}; height: 6px; border-radius: 3px;">
<div style="background: #4facfe; width: {prob}%; height: 100%; border-radius: 3px;"></div>
</div>
</div>"""

        # Removed indentation for the entire layout string
        st.markdown(f"""
<div class='glass-card'>
<h4 style='margin-top:0; font-weight:600; color:{colors["text"]}; font-size:18px; margin-bottom: 20px;'>Analysis Results</h4>
<div class='{res_class}' style='text-align: center; padding: 20px;'>
<div style='font-size: 24px;'><b>Result:</b> {res_text}</div>
<div style='font-size: 18px;'><b>Type:</b> {type_text}</div>
</div>
<div class="stat-container">
<div class="stat-box">
<div class="stat-value">{conf:.1f}%</div>
<div class="stat-label">Confidence</div>
</div>
<div class="stat-box">
<div class="stat-value">{MODEL_ACCURACY}</div>
<div class="stat-label">Model Accuracy</div>
</div>
</div>
<hr style="border: 0; border-top: 1px solid {colors['divider']}; margin: 25px 0;">
<h5 style='font-weight:600; color:{colors["text"]}; font-size:15px; margin-bottom: 15px;'>Probability Distribution</h5>
{bars_html}
</div>
""", unsafe_allow_html=True)