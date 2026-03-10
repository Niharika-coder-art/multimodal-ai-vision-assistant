# Multimodal AI Vision Assistant

A powerful Streamlit-based web application that provides multimodal AI capabilities including image captioning, visual question answering (VQA), and object detection.

## Features

- **📝 Image Captioning**: Generate descriptive captions for any uploaded image using BLIP model
- **🤖 Visual Question Answering (VQA)**: Ask questions about images and get AI-powered answers
- **🔍 Object Detection**: Detect and identify objects in images using YOLOv8

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)

## Project Structure

```
multimodal-vqa-system/
├── app.py                  # Main Streamlit application
├── models/
│   ├── caption_model.py   # Image captioning model
│   ├── vqa_model.py       # Visual Question Answering model
│   └── detection_model.py # Object detection model
├── images/                # Uploaded images directory
├── yolov8n.pt            # YOLO model weights
└── requirements.txt      # Python dependencies
```

## Usage

1. Run the Streamlit app
2. Upload an image (JPG, PNG, or JPEG)
3. Choose from three options:
   - Click "Generate Caption" to get an image description
   - Enter a question and click "Ask AI" to get answers
   - Click "Detect Objects" to identify objects in the image

## Optimizations

This version includes several performance optimizations:

- **Model Caching**: Models are cached using `@st.cache_resource` to avoid reloading
- **GPU Acceleration**: Automatic detection and use of CUDA when available
- **Efficient Inference**: Optimized tensor operations with proper device management
- **Error Handling**: Comprehensive try-except blocks for robust operation
- **Session State**: Results persist across interactions

## Models Used

- **BLIP**: Salesforce's Bootstrapped Language-Image Pre-training for VQA and Captioning
- **YOLOv8**: Ultralytics' state-of-the-art object detection model

## License

MIT License

