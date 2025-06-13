import streamlit as st
import requests
import cv2
import numpy as np
from PIL import Image
import time
import os
from ultralytics import YOLO  # Import YOLO from Ultralytics

# Configuration - Read from secrets
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", os.getenv("HF_API_TOKEN"))
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

# Load your trained YOLO model
@st.cache_resource
def load_detection_model():
    """Load your trained YOLO model from file or URL"""
    try:
        # Load from local file (preferred)
        model = YOLO('bantai_obj_det_model.pt')  # Replace with your model path
        
        # Alternative: Load from remote URL
        # model = YOLO('https://your-domain.com/models/ppe_detection.pt')
        
        return model
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {str(e)}")
        return None

# Object detection using YOLO
def detect_ppe(image_np, model):
    """
    Detect PPE using YOLO model
    Returns: (hairnet_detected: bool, gloves_detected: bool)
    """
    if model is None:
        return False, False
    
    # Run inference
    results = model.predict(image_np, conf=0.5)  # Adjust confidence threshold as needed
    
    # Initialize detection flags
    hairnet_detected = False
    gloves_detected = False
    
    # Process results
    for result in results:
        # Check for detected classes
        for cls in result.boxes.cls:
            class_id = int(cls)
            class_name = model.names[class_id]
            
            # Update detection flags based on class names
            if class_name == "hairnet":
                hairnet_detected = True
            elif class_name == "gloves":
                gloves_detected = True
    
    return hairnet_detected, gloves_detected

# Generate OSH recommendations using Hugging Face API
def generate_osh_recommendation(violations):
    # ... (unchanged from your previous implementation) ...

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

# Load model once at startup
ppe_model = load_detection_model()

# File Upload Section
with st.expander("📤 Upload Media", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload workplace image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi"],
        help="Supported formats: Images (JPG, PNG), Videos (MP4, AVI)"
    )

# Processing Section
if uploaded_file:
    # Image Processing
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Workplace Image", use_column_width=True)
        
        if st.button("🔍 Analyze Compliance", type="primary"):
            if ppe_model is None:
                st.error("Detection model not loaded. Unable to process.")
                st.stop()
                
            with st.spinner("Detecting PPE compliance..."):
                # Convert to OpenCV format
                img_np = np.array(image)
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                # Run detection
                hairnet, gloves = detect_ppe(img_np, ppe_model)
                
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
    
    # Video Processing
    elif uploaded_file.type.startswith("video"):
        st.info("Video processing started. This may take several minutes...")
        
        # Save video to temp file
        temp_video = f"temp_{uploaded_file.name}"
        with open(temp_video, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize analysis
        hairnet_frames = 0
        gloves_frames = 0
        total_frames = 0
        violation_frames = []
        
        # Display placeholder for processing updates
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Process video
        cap = cv2.VideoCapture(temp_video)
        frame_skip = 5  # Process every 5th frame to reduce computation
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Process only every nth frame
            if total_frames % frame_skip != 0:
                continue
                
            # Update progress
            progress = min(int((cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100), 100)
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {total_frames}...")
            
            # Detect PPE
            hairnet, gloves = detect_ppe(frame, ppe_model)
            
            if hairnet: hairnet_frames += 1
            if gloves: gloves_frames += 1
            
            if not hairnet or not gloves:
                violation_frames.append(total_frames)
        
        # Release resources
        cap.release()
        os.remove(temp_video)
        
        # Calculate compliance rates
        processed_frames = total_frames // frame_skip
        hairnet_rate = (hairnet_frames / processed_frames) * 100 if processed_frames > 0 else 0
        gloves_rate = (gloves_frames / processed_frames) * 100 if processed_frames > 0 else 0
        
        # Display results
        st.success("✅ Video analysis complete!")
        col1, col2 = st.columns(2)
        col1.metric("Hairnet Compliance", f"{hairnet_rate:.1f}%", 
                   delta_color="inverse", 
                   help="Percentage of frames with hairnet detected")
        col2.metric("Gloves Compliance", f"{gloves_rate:.1f}%", 
                   delta_color="inverse",
                   help="Percentage of frames with gloves detected")
        
        # Generate recommendations
        violations = []
        if hairnet_rate < 95: violations.append("inconsistent hairnet usage")
        if gloves_rate < 95: violations.append("inconsistent glove usage")
        
        st.divider()
        st.subheader("📜 OSH Compliance Report (Philippines)")
        
        with st.spinner("Generating legal recommendations..."):
            report = generate_osh_recommendation(violations)
            st.markdown(report)
        
        # Show violation frames
        if violation_frames:
            st.subheader("⚠️ Violation Frames Detected")
            st.write(f"PPE violations detected in {len(violation_frames)} frames:")
            st.write(violation_frames[:20])  # Show first 20 frame numbers

# ... (rest of the code remains unchanged: Resources, Footer, etc.) ...
