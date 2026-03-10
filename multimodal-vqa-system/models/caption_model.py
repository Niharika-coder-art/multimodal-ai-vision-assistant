import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import streamlit as st

@st.cache_resource
def load_caption_model():
    """Load caption model with caching for performance"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    model.eval()
    return processor, model, device

def generate_caption(image_path):
    """Generate caption for the given image"""
    try:
        processor, model, device = load_caption_model()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs)
        
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

