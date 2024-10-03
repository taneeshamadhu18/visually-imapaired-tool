import streamlit as st
import cv2
import numpy as np
import pytesseract
import pyttsx3
from PIL import Image
import torch
from transformers import pipeline


engine = pyttsx3.init()


ner_pipelines = {
    "English": pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple"),
    "Hindi": pipeline("ner", model="ai4bharat/indic-bert", aggregation_strategy="simple"),
    "Kannada": pipeline("ner", model="ai4bharat/indic-bert", aggregation_strategy="simple")
}


def set_voice(lang):
    voices = engine.getProperty('voices')
    for voice in voices:
        if lang.lower() in voice.languages:
            engine.setProperty('voice', voice.id)
            break

def analyze_image(image, lang):
  
    img = np.array(image)

    
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s') 
    results = model(img)

    OBJECT_LABELS = [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
        'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

   
    description = []

    detections = results.pred[0] 
    if detections is not None and len(detections) > 0:
        for *box, conf, cls in detections:
            label = model.names[int(cls)]
            if label in OBJECT_LABELS:
                x1, y1, x2, y2 = map(int, box)
                detected_text = f"Detected {label} with confidence {conf:.2f} at coordinates: ({x1}, {y1}, {x2}, {y2})"
                description.append(detected_text)
               
                speak_text(f"Detected {label} with confidence {conf:.2f}.", lang)
    else:
        description.append("No objects detected.")

    
    ocr_text = perform_ocr(img)
    description.append("Detected Text: " + ocr_text)

   
    nlp_analysis = analyze_text(ocr_text, lang)
    description.append("NLP Analysis: " + nlp_analysis)

    return " ".join(description)

def perform_ocr(image):
   
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = pytesseract.image_to_string(image_rgb)
    return text.strip()

def analyze_text(text, lang):
    if not text:
        return "No text detected for analysis."

   
    entities = ner_pipelines[lang](text)
    
    if entities:
        entity_descriptions = [
            f"{ent['word']} ({ent['entity_group']})" for ent in entities
        ]
        return f"Detected entities: {', '.join(entity_descriptions)}."
    else:
        return "No entities detected."

def speak_text(text, lang):
    set_voice(lang)  
    if text:
        engine.say(text)
        engine.runAndWait()


st.title("Accessibility Tool for the Visually Impaired")


language = st.selectbox("Select Language", ["English", "Hindi", "Kannada"])


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
       
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        
        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                description = analyze_image(image, language)
                st.write("Analysis Result:")
                st.write(description)

                
                speak_text(description, language)
    except Exception as e:
        st.error(f"Error processing the image: {e}")


st.header("Text to Speech")
text_input = st.text_input("Enter text to speak:")
if st.button("Speak"):
    speak_text(text_input, language)
    st.success(f"Speaking: {text_input}")
