import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from PIL import Image
from pathlib import Path

st.title('Corn Plant Diseases Classifier')

body = """
### The corn plant diseases are classified into four different classes: 
1. Healthy 
2. Northern Leaf Blight 
3. Common Rust 
4. Cercospora Leaf Spot (Gray Leaf Spot)
"""

def predict(image):
    IMAGE_RES = (260, 260)
    corn_classifier_model = 'saved-model-b2/best_model'
    model = load_model(corn_classifier_model, compile=False)
    classes = ['Cercospora Leaf Spot (Gray Leaf Spot)', 'Common Rust', 'Northern Leaf Blight', 'Healthy']
    
    test_images = image.resize(IMAGE_RES)
    test_images = preprocessing.image.img_to_array(test_images)
    test_images = test_images / 255.0
    test_images = np.expand_dims(test_images, axis=0)
    print(test_images)

    predictions = model.predict(test_images)
    scores = tf.nn.softmax(predictions[0])
    print(scores)
    scores = scores.numpy()
    print(scores)
    results = {
        'Cercospora Leaf Spot (Gray Leaf Spot)': 0,
        'Common Rust': 0,
        'Northern Leaf Blight': 0,
        'Healthy': 0,
    }

    results = f"{classes[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2)} % confidence score."
    return results


def main():
    nav_select = st.sidebar.selectbox('Please select from the following', ['Classifier', 'Model Statistics and Review'])

    if nav_select == 'Model Statistics and Review':
        st.sidebar.success('Information are now shown on the right for desktop users.')
        st.markdown(Path('./Thesis_info.md').read_text())

    if nav_select == 'Classifier':
        st.sidebar.success('You can now start classifying!')
        st.markdown(body)
        file_upload = st.file_uploader('Select an Image', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
        submit_button = st.button('Classify')
        if file_upload is not None:
            for img in file_upload:
                image = Image.open(img)
                st.image(image, caption='Image Preview', use_column_width=True)
                if submit_button:
                    with st.spinner('Ongoing Classification...'):
                        plt.imshow(image)
                        plt.axis('off')
                        predictions = predict(image)
                        time.sleep(1)
                    st.success('Image Has Been Classified')
                    st.write(predictions)

if __name__ == '__main__':
    main()