import torch
from ultralytics import YOLO
import streamlit as st
from PIL import Image
import numpy as np

@st.cache_resource
def load_detection_model():
    """Load YOLO detection model with caching for performance"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8n.pt")
    model.to(device)
    return model, device

def detect_objects(image_path):
    """Detect objects in the given image and return annotated image"""
    try:
        model, device = load_detection_model()
        results = model(image_path, device=device)
        
        # Get annotated image (numpy array)
        img_array = results[0].plot()
        
        # Convert to PIL Image for Streamlit compatibility
        img = Image.fromarray(img_array)
        
        return img
    except Exception as e:
        st.error(f"Error detecting objects: {str(e)}")
        return None

