import os
import random
import streamlit as st 
from ultralytics import YOLO
import cv2
import time
import numpy as np
import shutil
#import google.generativeai as palm
import google.generativeai as genai
from googletrans import Translator


@st.cache_resource()    
def main_model():
    model=YOLO('best.pt')

    return model

        
def message():
    st.warning('⚠️Please check your image')
    st.info("📷✨ **Encountering the 'Please check your image' error?**")
    st.write("""
            Our algorithm may not have been able to predict the content of your image. To improve results, consider the following:
            👉 **Verify image quality and resolution.**
            👉 **Ensure the image is clear and well-lit.**
            👉 **Check if the image meets our specified format requirements.**
            👉 **Consider alternative images for better results.**
            Our aim is to provide accurate predictions, and addressing these aspects can make a significant difference. If the issue persists, please reach out to our support team. We're here to help! 🤝🔧
            """)

def upload():
    image=None
    image_filename=None
    initial_image = st.camera_input('Take a picture')
    original_image = initial_image
    temp_path = None
    if initial_image is not None:
        image_filename = f"{int(time.time())}.jpg"
        bytes_data = initial_image.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
    return image, original_image,image_filename
 
def process_image_with_yolo(pic0):
    names={
                0: 'beet',
                1: 'bell_pepper',
                2: 'cabbage',
                3: 'carrot',
                4: 'cucumber',
                5: 'egg',
                6: 'eggplant',
                7: 'garlic',
                8: 'onion',
                9: 'potato',
                10: 'tomato',
                11: 'zucchini'
            
            }
        
    
    labelslist=[]
    # Load your YOLO model
    
    if pic0 is not None:
        # Perform YOLO prediction on the image
        
        model = main_model()
        
       
        pic0=pic0
        result = model.predict(pic0,conf=0.8, save=True, save_txt=True)
        
        txt_files_exist = any(filename.endswith('.txt') for filename in os.listdir('runs/detect/predict/labels'))

        if txt_files_exist:
            
            lis = open('runs/detect/predict/labels/image0.txt', 'r').readlines()
            for line in lis:
                bresults=line.split(" ")
                bresults=int(bresults[0])
                clabel=names[bresults]
                labelslist.append(clabel)
                #process_line(line, image_np2)
                
          
           
        labels_count = {}
        for label in labelslist:
            if label in labels_count:
                labels_count[label] += 1
            else:
                labels_count[label] = 1
        labelslist=[]
        
        try:
            if os.path.exists('runs'):
                shutil.rmtree('runs')
                st.session_state.original_image = None  # Clear the original_image variable
                           
        except Exception as e:
            st.error(f"An error occurred: {e}")
        
              
    
    return labels_count
        
                
                
                
                
                
                
                
def generate_recipe(vegetable_dict, target_lang,recipe):
    genai.configure(api_key=st.secrets['key'])
    #palm.set_random_seed()

    prompt = f"Create nutritional {recipe} delightful and concise recipes using the following vegetables. Each recipe should include a dish name, a list of ingredients, detailed cooking instructions along with the nuritritional values serving per (cup). Numbered each recipe\n\nIngredients:\n"

    for vegetable, count in vegetable_dict.items():
        prompt += f"- {vegetable} ({count} {'piece' if count == 1 else 'pieces'})\n"

    prompt += "\nYour recipes should only use the mentioned vegetables. Be creative, and make the instructions clear and easy to follow. These recipes should be suitable for anyone looking to enjoy quick and tasty dishes."
    
    model=genai.GenerativeModel('gemini-pro')
    res = model.generate_content(prompt)
    gt = res.text.replace('*', '')  # Remove asterisk marks

    # Translate the generated text to the target language
    translator = Translator()
    translated_text = translator.translate(gt, src='en', dest=target_lang)
    
    return translated_text.text
    
