import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf


@st.cache_resource
def load_model_and_labels():
    
    model = tf.keras.models.load_model('emoji_classifier_model.h5')
    
    
    label_classes = np.load('label_classes.npy', allow_pickle=True)
    
    return model, label_classes


def predict_emoji_name(image):
    
    image = load_img(image, target_size=(64, 64 ))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  
    
   
    model, label_classes = load_model_and_labels()

    
    predictions = model.predict(image)
    predicted_label = np.argmax(predictions, axis=1)

    
    emoji_name = label_classes[predicted_label][0]
    
    return emoji_name

st.title("Emoji Prediction App")

st.write("Upload an emoji image and get the predicted emoji name.")


uploaded_file = st.file_uploader("Choose an emoji image...", type="png")

if uploaded_file is not None:
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    emoji_name = predict_emoji_name(uploaded_file)
    st.write(f"The predicted emoji name is: {emoji_name}")
