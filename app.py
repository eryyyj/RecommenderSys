import os
import time
import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
import streamlit as st

st.set_page_config(
    page_title="PPE Compliance Inspector",
    page_icon="🛡️",
    layout="wide"
)

HF_API_TOKEN = st.secrets["secrets"]["HF_API_TOKEN"].strip()
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('bantai_obj_det_model.pt')
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading YOLO model: {str(e)}")
        return None

# Load model at startup
yolo_model = load_yolo_model()

# Object detection function using YOLO
def detect_ppe(image_np):
    """
    Detect PPE using YOLO model
    Returns: (hairnet_detected: bool, gloves_detected: bool)
    """
    if yolo_model is None:
        return False, False
    
    # Run inference
    results = yolo_model.predict(image_np, conf=0.5)  # Adjust confidence as needed
    
    # Initialize detection flags
    hairnet_detected = False
    gloves_detected = False
    
    # Process results
    for result in results:
        # Check for detected classes
        for box in result.boxes:
            class_id = int(box.cls)
            class_name = yolo_model.names[class_id]
            
            # Update detection flags based on class names
            if class_name == "hairnet":
                hairnet_detected = True
            elif class_name == "gloves":
                gloves_detected = True
    
    return hairnet_detected, gloves_detected

# Generate OSH recommendations using Hugging Face API
def generate_osh_recommendation(violations):
    if not violations:
        return "✅ All PPE requirements met according to Philippine OSH standards."
    
    violations_text = " and ".join(violations)
    
    # Updated prompt for better compatibility
    prompt = f"""
    As an OSH compliance officer in the Philippines, create a formal report for PPE violations: {violations_text}.
    
    Structure your response:
    1. Relevant Philippine OSH Law sections (cite RA 11058 and DOLE D.O. 198-18 specifically)
    2. Potential penalties under Philippine law
    3. 3 immediate corrective actions
    4. Preventive measures
    5. Training recommendations
    
    Use formal tone and focus exclusively on Philippine legal context.
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
            timeout=120
        )
        
        if response.status_code == 404:
            return "⚠️ Model not available via API. Try a different model in code configuration."
        
        if response.status_code != 200:
            return f"⚠️ API Error ({response.status_code}): {response.text[:200]}..."
            
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            # Handle different response formats
            if 'generated_text' in result[0]:
                return result[0]['generated_text'].strip()
            elif 'text' in result[0]:
                return result[0]['text'].strip()
        
        return f"⚠️ Unexpected response format: {str(result)[:300]}"
    
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# Generate chat responses using Hugging Face API
def generate_chat_response(user_input):
    """
    Generate a response to user chat input using the LLM model
    """
    if not user_input or user_input.strip() == "":
        return "Please enter a question about Philippine OSH laws."
    
    # System prompt to guide the assistant
    system_prompt = "You are an AI assistant specialized in Occupational Safety and Health (OSH) laws in the Philippines. Provide accurate, concise, and helpful information based on the Philippine OSH standards (RA 11058, DOLE D.O. 198-18)."
    
    # Format the prompt for the model
    prompt = f"""
    <|system|>
    {system_prompt}</s>
    <|user|>
    {user_input}</s>
    <|assistant|>
    """
    
    if not HF_API_TOKEN:
        return "⚠️ Error: Hugging Face API token not configured."
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.6,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }
    
    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{MODEL_NAME}",
            headers=headers,
            json=payload,
            timeout=90
        )
        
        if response.status_code != 200:
            return f"⚠️ API Error ({response.status_code}): {response.text[:200]}"
            
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            if 'generated_text' in result[0]:
                # Extract only the assistant's response
                full_response = result[0]['generated_text']
                # Find the start of the assistant's response
                start_idx = full_response.find("<|assistant|>") + len("<|assistant|>")
                return full_response[start_idx:].strip()
        
        return "⚠️ Sorry, I couldn't generate a response. Please try again."
    
    except Exception as e:
        return f"⚠️ API Error: {str(e)}"

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main App
st.title("🛡️ PPE Compliance Inspector")
st.subheader("Philippine Occupational Safety and Health (OSH) Compliance System")
st.markdown("""
    *Detect hairnet and glove violations in workplace environments and generate compliance reports based on Philippine laws*
""")

# File Upload Section
with st.expander("📤 Upload Media", expanded=True):
    uploaded_file = st.file_uploader(
        "Upload workplace image or video",
        type=["jpg", "jpeg", "png", "mp4"],
        help="Supported formats: Images (JPG, PNG), Videos (MP4)"
    )

# Processing Section
if uploaded_file:
    # Image Processing
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Workplace Image", use_column_width=True)
        
        if st.button("🔍 Analyze Compliance", type="primary"):
            if yolo_model is None:
                st.error("YOLO model not loaded. Unable to process.")
                st.stop()
                
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
    
    # Video Processing
    elif uploaded_file.type.startswith("video"):
        if yolo_model is None:
            st.error("YOLO model not loaded. Unable to process video.")
            st.stop()
            
        st.info("Video processing started. This may take several minutes...")
        
        # Save video to temp file
        temp_video = f"temp_{uploaded_file.name}"
        with open(temp_video, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Initialize analysis
        hairnet_frames = 0
        gloves_frames = 0
        total_frames = 0
        processed_frames = 0
        
        # Display placeholder for processing updates
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        # Process video
        cap = cv2.VideoCapture(temp_video)
        frame_skip = 5  # Process every 5th frame to reduce computation
        
        # Create a placeholder for the video preview
        video_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Process only every nth frame
            if total_frames % frame_skip != 0:
                continue
                
            processed_frames += 1
                
            # Update progress
            progress = min(int((cap.get(cv2.CAP_PROP_POS_FRAMES) / cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100), 100))
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {total_frames}...")
            
            # Detect PPE
            hairnet, gloves = detect_ppe(frame)
            
            if hairnet: hairnet_frames += 1
            if gloves: gloves_frames += 1
            
            # Display preview every 50 frames
            if processed_frames % 50 == 0:
                # Convert back to RGB for display
                preview_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(preview_frame, caption="Processing Preview", width=600)
        
        # Release resources
        cap.release()
        os.remove(temp_video)
        
        # Calculate compliance rates
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

# Chatbot Section
st.divider()
st.subheader("💬 OSH Law Chat Assistant")

# Display chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask about Philippine OSH laws...")

if user_input:
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("▌")
        
        with st.spinner("Thinking..."):
            response = generate_chat_response(user_input)
        
        message_placeholder.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Clear chat button
if st.button("Clear Chat History", key="clear_chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()

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

# Chat tips
st.sidebar.divider()
st.sidebar.subheader("💡 Chat Assistant Tips")
st.sidebar.markdown("""
    Ask about:
    - Hairnet requirements under RA 11058
    - Penalties for PPE violations
    - OSH training requirements
    - Glove specifications for food handling
    - DOLE inspection procedures
""")

# Deployment Info
st.sidebar.divider()
if st.sidebar.button("🔄 Check Deployment Status"):
    try:
        import streamlit as st_module
        st.sidebar.success(f"Streamlit v{st_module.__version__}")
        st.sidebar.info(f"Python {os.sys.version}")
        if yolo_model:
            st.sidebar.success("YOLO model loaded successfully")
        else:
            st.sidebar.error("YOLO model not loaded")
    except:
        st.sidebar.error("Version info unavailable")

# Footer
st.divider()
st.caption("""
    *Disclaimer: This application provides general guidance only. Recommendations are AI-generated 
    and should be verified by qualified OSH professionals. Always refer to the latest DOLE regulations.*
""")
st.caption(f"App version: 2.0 | Last updated: {time.strftime('%Y-%m-%d')}")
