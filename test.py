import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf

# Load your model
MODEL = tf.keras.models.load_model("mymodel.h5")
CLASS_NAMES = ['Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato__Target_Spot',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = np.expand_dims(img, axis=0)
    return img


# Streamlit UI code
st.title('ML Model Image Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make predictions on the uploaded image
    if st.button('Predict'):
        img_array = preprocess_image(uploaded_file)
        predictions = MODEL.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        print(predictions[0])
        confidence = np.max(predictions[0]) * 100  # Convert to percentage
        st.success(f'Prediction: {predicted_class}, Confidence: {confidence:.2f}%')
