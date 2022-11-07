import streamlit as st
import tensorflow as tf
import streamlit as st
import pathlib
from PIL import Image
from numpy import asarray

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('C:\LCA Attempt\my_model2.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Image Dehazing
         """
         )
file = st.file_uploader("Please upload an image to be dehazed", type=["png"])


#txt = st.text_input(label="directory")
import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib.image import imread
import matplotlib.pyplot as plt
import os

st.set_option('deprecation.showfileUploaderEncoding', False)
def get_img(image,model):
    image = Image.open(file)
    
    image = np.asarray(image)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255.0
    image  = image.reshape(1, 460, 620, 3)
    img = model.predict(image)
    img = img*255.0
    img = img.astype('uint8')
    img = img.reshape(460,620,3)
    return img
from PIL import Image, ImageOps      
if file is None:
    st.text("Please upload an image file")
else:
    #img2 = Image.open(file)
    #img2 = st.image(img2, use_column_width=True,output_format='png')
    st.write("Before")
    file2 = Image.open(file)
    file2 = file2.resize((620,460))
    #file2  = file2.convert("P", palette=Image.ADAPTIVE, colors=24)
    st.image(file2)
    
    #file2.thumbnail((620,460))
    #file2 = Image.open(file2)
    
    predictions = get_img(file2,model)
    #predictions = model.predict(img32)
    #img2 = st.image(img2, use_column_width=True,output_format='png')
    st.write('after')
    st.image(predictions, use_column_width=True, clamp=True,channels="RGB")
    
  

