# Multimodal AI Vision Assistant

An advanced **Multimodal AI System** that can understand images and answer questions about them using deep learning models.

This project combines **Computer Vision + Natural Language Processing** to build an intelligent visual assistant.

---

## Features

* Visual Question Answering (Ask questions about an image)
* Image Caption Generation
* Object Detection using YOLOv8
* Streamlit Web Interface
* Multimodal AI (Image + Text Understanding)

---

## Technologies Used

* Python
* Streamlit
* HuggingFace Transformers
* BLIP (Vision Language Model)
* YOLOv8 (Object Detection)
* PyTorch
* Pillow

---

## Project Architecture

Image Input → Vision Model → Language Model → AI Answer

Models used:

* BLIP Image Captioning
* BLIP Visual Question Answering
* YOLOv8 Object Detection

---

## Project Structure

```
multimodal-ai-vision-assistant
│
├── app.py
├── requirements.txt
├── README.md
│
├── images
│   └── input.jpg
│
├── models
│
└── utils
```

---

## Installation

Clone the repository

```
git clone https://github.com/your-username/multimodal-ai-vision-assistant.git
```

Go to the project folder

```
cd multimodal-ai-vision-assistant
```

Install dependencies

```
pip install -r requirements.txt
```

---

## Run the Application

```
streamlit run app.py
```

Then open in browser:

```
http://localhost:8501
```

---

## Example Use Cases

Upload an image and ask questions like:

* What is in this image?
* How many people are there?
* What objects are present?

The system analyzes the image and generates intelligent answers.

---

## Applications

* AI Assistants
* Visual Search Systems
* Smart Surveillance
* Accessibility Tools for Visually Impaired
* AI-powered Image Understanding

---

## Future Improvements

* Voice Question Input
* Real-time camera analysis
* OCR text extraction
* Multilingual support
* ChatGPT-style conversational AI

---

## Author

Developed by **Niharika Chinthada**

B.Tech Artificial Intelligence & Machine Learning

---

## License

This project is open-source and available under the MIT License.
