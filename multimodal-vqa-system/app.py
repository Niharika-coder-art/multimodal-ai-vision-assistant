import streamlit as st
import os
from PIL import Image
import numpy as np

# Import model functions
from models.caption_model import generate_caption
from models.vqa_model import answer_question
from models.detection_model import detect_objects

# Page configuration
st.set_page_config(
    page_title="Multimodal AI Vision Assistant",
    page_icon="👁️",
    layout="wide"
)

# Ensure images directory exists
IMAGE_DIR = "images"
os.makedirs(IMAGE_DIR, exist_ok=True)

st.title("👁️ Multimodal AI Vision Assistant")
st.markdown("Upload an image to generate captions, ask questions, or detect objects!")

# Initialize session state for storing results
if 'caption' not in st.session_state:
    st.session_state.caption = None
if 'answer' not in st.session_state:
    st.session_state.answer = None
if 'detection_img' not in st.session_state:
    st.session_state.detection_img = None

# File uploader
uploaded_file = st.file_uploader(
    "Upload an image (JPG, PNG, JPEG)", 
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    # Save uploaded image
    image_path = os.path.join(IMAGE_DIR, "input.jpg")
    
    try:
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Uploaded Image")
            st.image(image_path, use_container_width=True)
        
        # Button actions in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📝 Generate Caption", use_container_width=True, key="caption_btn"):
                with st.spinner("Generating caption..."):
                    st.session_state.caption = generate_caption(image_path)
        
        with col2:
            question = st.text_input(
                "Ask a question about the image", 
                placeholder="e.g., What color is the car?",
                key="question_input"
            )
            if st.button("🤖 Ask AI", use_container_width=True, key="ask_btn"):
                if question.strip():
                    with st.spinner("Analyzing image..."):
                        st.session_state.answer = answer_question(image_path, question)
                else:
                    st.warning("Please enter a question!")
        
        with col3:
            if st.button("🔍 Detect Objects", use_container_width=True, key="detect_btn"):
                with st.spinner("Detecting objects..."):
                    st.session_state.detection_img = detect_objects(image_path)
        
        # Display results
        st.divider()
        st.subheader("Results")
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if st.session_state.caption:
                st.success("Caption Generated!")
                st.write(f"**{st.session_state.caption}**")
        
        with result_col2:
            if st.session_state.answer:
                st.success("Answer:")
                st.write(f"**{st.session_state.answer}**")
        
        # Display detection results
        if st.session_state.detection_img is not None:
            st.subheader("Object Detection Results")
            st.image(st.session_state.detection_img, caption="Detected Objects", use_container_width=True)
            
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
else:
    # Show placeholder when no image uploaded
    st.info("👆 Please upload an image to get started!")
    
    # Show sample usage
    st.markdown("""
    ### Features:
    - **📝 Generate Caption**: Get AI-generated description of your image
    - **🤖 Ask AI**: Ask questions about objects in the image
    - **🔍 Detect Objects**: Identify and locate objects using YOLO
    """)

# Add footer
st.markdown("---")
st.markdown("*Powered by BLIP and YOLOv8*")

