import os
import random
import streamlit as st 
from ultralytics import YOLO
import cv2
import time
import numpy as np
import shutil
import google.generativeai as palm
from googletrans import Translator
'''
image_directory = "val"  # Assuming "val" is the directory name

# Get a list of image filenames in the directory
image_filenames = [filename for filename in os.listdir(image_directory) if filename.endswith(".jpg")]

# Function to generate a random image from the list of filenames
def get_random_image():
    if not image_filenames:
        return None
    random_image_filename = random.choice(image_filenames)
    random_image_filename1=random_image_filename.split('.')[0]
    random_image_path = os.path.join(image_directory, random_image_filename)
    return random_image_path,random_image_filename1 
'''

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
        #hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        #value = 42 #whatever value you want to add
        #cv2.add(hsv[:,:,2], value, hsv[:,:,2])
        #image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        #image=cv2.multiply(image,0.00005)
    return image, original_image,image_filename
 
def process_image_with_yolo(pic0):
    names={
            0: 'avocado',
            1: 'beans',
            2: 'beet',
            3: 'bell pepper',
            4: 'broccoli',
            5: 'brus capusta',
            6: 'cabbage',
            7: 'carrot',
            8: 'cayliflower',
            9: 'celery',
            10: 'corn',
            11: 'cucumber',
            12: 'eggplant',
            13: 'fasol',
            14: 'garlic',
            15: 'hot pepper',
            16: 'onion',
            17: 'peas',
            18: 'potato',
            19: 'pumpkin',
            20: 'rediska',
            21: 'redka',
            22: 'salad',
            23: 'squash-patisson',
            24: 'tomato',
            25: 'vegetable marrow'
            
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
    palm.configure(api_key=st.secrets['key'])
    #palm.set_random_seed()
    prompt = f"Create {recipe} delightful and concise recipes using the following vegetables. Each recipe should include a dish name, a list of ingredients, and cooking instructions. Numbered each recipe\n\nIngredients:\n"

    for vegetable, count in vegetable_dict.items():
        prompt += f"- {vegetable} ({count} {'piece' if count == 1 else 'pieces'})\n"

    prompt += "\nYour recipes should only use the mentioned vegetables. Be creative, and make the instructions clear and easy to follow. These recipes should be suitable for anyone looking to enjoy quick and tasty dishes."
  

    res = palm.generate_text(prompt=prompt)
    gt = res.result.replace('*', '')  # Remove asterisk marks

    # Translate the generated text to the target language
    translator = Translator()
    translated_text = translator.translate(gt, src='en', dest=target_lang)
    
    return translated_text.text
    
