import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import streamlit as st

@st.cache_resource
def load_vqa_model():
    """Load VQA model with caching for performance"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model.to(device)
    model.eval()
    return processor, model, device

def answer_question(image_path, question):
    """Answer a question about the given image"""
    try:
        processor, model, device = load_vqa_model()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, question, return_tensors="pt").to(device)
        
        with torch.no_grad():
            out = model.generate(**inputs)
        
        answer = processor.decode(out[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        return f"Error answering question: {str(e)}"

