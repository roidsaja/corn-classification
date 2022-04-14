import streamlit as st

st.title('Corn Plant Diseases Classifier')

body = """
### The corn plant diseases are classified into four different classes: 
1. Healthy 
2. Northern Leaf Blight 
3. Common Rust 
4. Cercospora Leaf Spot (Gray Leaf Spot)
"""
st.markdown(body)

def main():
    file_upload = st.file_uploader('Select an Image', type=['jpg', 'jpeg', 'png'])
    submit_button = st.button('Upload')

if __name__ == '__main__':
    main()