import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import time
import os

# Configuration - Read from secrets
HF_API_TOKEN = HF_API_TOKEN = st.secrets["secrets"]["HF_API_TOKEN"].strip()
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

# Placeholder for object detection
def detect_ppe(image_np):
    """
    Simulated detection function - replace with your actual model
    Returns: (hairnet_detected: bool, gloves_detected: bool)
    """
    # Convert to HSV for color-based simulation
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)
    
    # Simple color-based detection simulation
    hairnet_detected = np.mean(hsv[:, :, 1]) > 50  # Random condition
    gloves_detected = np.mean(hsv[:, :, 2]) > 100   # Random condition
    
    return hairnet_detected, gloves_detected

# Generate OSH recommendations using Hugging Face API
def generate_osh_recommendation(violations):
    if not violations:
        return "✅ All PPE requirements met according to Philippine OSH standards."
    
    violations_text = " and ".join(violations)
    
    prompt = f"""
    [INST] As an OSH compliance officer in the Philippines, create a formal report for PPE violations: {violations_text}.
    
    Structure your response:
    1. Relevant Philippine OSH Law sections (cite RA 11058 and DOLE D.O. 198-18 specifically)
    2. Potential penalties under Philippine law
    3. 3 immediate corrective actions
    4. Preventive measures
    5. Training recommendations
    
    Use formal tone and focus exclusively on Philippine legal context. [/INST]
    """
    
    if not HF_API_TOKEN:
        return "⚠️ Error: Hugging Face API token not configured. Please set HF_API_TOKEN in secrets."
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.3,
            "top_p": 0.9
        }
    }
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json=payload,
            timeout=120  # Longer timeout for Hugging Face
        )
        
        if response.status_code != 200:
            return f"⚠️ API Error ({response.status_code}): {response.text}"
            
        result = response.json()
        
        if isinstance(result, dict) and 'error' in result:
            return f"⚠️ Model Error: {result['error']}"
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '').split('[/INST]')[-1].strip()
        
        return "⚠️ Unexpected response format from API"
    
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# Streamlit App Configuration
st.set_page_config(
    page_title="PPE Compliance Inspector",
    page_icon="🛡️",
    layout="wide"
)

# Main App
st.title("🛡️ PPE Compliance Inspector")
st.subheader("Philippine Occupational Safety and Health (OSH) Compliance System")
st.markdown("""
    *Detect hairnet and glove violations in workplace environments and generate compliance reports based on Philippine laws*
""")

# File Upload Section
with st.expander("📤 Upload Media", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload workplace image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, PNG"
    )

# Processing Section
if uploaded_file and uploaded_file.type.startswith("image"):
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Workplace Image", use_column_width=True)
    
    if st.button("🔍 Analyze Compliance", type="primary"):
        with st.spinner("Detecting PPE compliance..."):
            # Convert to OpenCV format
            img_np = np.array(image)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Run detection
            hairnet, gloves = detect_ppe(img_np)
            
            # Display results
            col1, col2 = st.columns(2)
            col1.metric("Hairnet Compliance", 
                        "✅ COMPLIANT" if hairnet else "❌ VIOLATION", 
                        delta=None,
                        help="Detects proper hairnet usage")
            col2.metric("Gloves Compliance", 
                        "✅ COMPLIANT" if gloves else "❌ VIOLATION", 
                        delta=None,
                        help="Detects proper glove usage")
            
            # Generate recommendations
            violations = []
            if not hairnet: violations.append("no hairnet")
            if not gloves: violations.append("no gloves")
            
            st.divider()
            st.subheader("📜 OSH Compliance Report (Philippines)")
            
            with st.spinner("Generating legal recommendations..."):
                report = generate_osh_recommendation(violations)
                st.markdown(report)

# Video Processing Section (Placeholder)
st.divider()
st.subheader("Video Processing")
st.warning("Video processing requires additional implementation and resources")
st.info("""
    For video processing capabilities:
    1. Upgrade to Streamlit Premium for more resources
    2. Implement frame-by-frame processing
    3. Use cloud-based video processing services
""")

# Resources Section
st.sidebar.title("🇵🇭 Philippine OSH Resources")
st.sidebar.markdown("""
    **Official References:**
    - [Republic Act 11058 (OSH Law)](https://lawphil.net/statutes/repacts/ra2018/ra_11058_2018.html)
    - [DOLE D.O. 198-18 (OSH Standards)](https://bwc.dole.gov.ph/)
    - [PPE Requirements Guide](https://www.oshc.dole.gov.ph/)
    
    **Compliance Tools:**
    - [OSH Checklist Generator](https://www.oshc.dole.gov.ph/tools)
    - [Training Module Download](https://www.oshc.dole.gov.ph/training-modules)
""")

st.sidebar.divider()
st.sidebar.markdown("""
    **About This App:**
    This tool provides AI-powered PPE compliance analysis. 
    Recommendations are generated based on Philippine OSH laws.
    
    *Note: For formal compliance assessments, consult a certified OSH practitioner.*
""")

# Deployment Info
st.sidebar.divider()
if st.sidebar.button("🔄 Check Deployment Status"):
    try:
        import streamlit as st_module
        st.sidebar.success(f"Streamlit v{st_module.__version__}")
        st.sidebar.info(f"Python {os.sys.version}")
    except:
        st.sidebar.error("Version info unavailable")

# Footer
st.divider()
st.caption("""
    *Disclaimer: This application provides general guidance only. Recommendations are AI-generated 
    and should be verified by qualified OSH professionals. Always refer to the latest DOLE regulations.*
""")
st.caption(f"App version: 1.0 | Last updated: {time.strftime('%Y-%m-%d')}")
