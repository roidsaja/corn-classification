import numpy as np
import tensorflow as tf
import time
import requests
import streamlit as st

from io import BytesIO
from tensorflow.keras.models import load_model
from PIL import Image


body = """
#### The corn plant diseases are classified into four different classes: 
1. Healthy 
2. Northern Leaf Blight 
3. Common Rust 
4. Cercospora Leaf Spot (Gray Leaf Spot)

Each image will have its own classification process and will include descriptions about 
the disease's facts, symptoms and strategies to handle them. The sources to these descriptions were 
all extracted from [CALS Cornell website](https://cals.cornell.edu/field-crops/corn/diseases-corn).

----

To get you started, a stock corn plant image has been reserved for you in the text field below. You can try out some other images too such as these:

- [Gray Leaf Spot](https://cropwatch.unl.edu/2018-CW-News/2018-images/Disease/corn-gray-leaf-spot-backlit-2.jpg)
- [Norther Leaf Blight](https://cropwatch.unl.edu/2018-CW-News/2018-images/Disease/corn-northern-corn-leaf-blight.jpg)

Image sources : [Cropwatch UNL website](https://cropwatch.unl.edu/2018/differentiating-corn-leaf-diseases)
"""

@st.experimental_memo(suppress_st_warning=True)
def predict(_image):
    IMAGE_RES = (224, 224)
    model = load_model('saved-model/best_model')
    classes = ['Cercospora Leaf Spot (Gray Leaf Spot)', 'Common Rust', 'Healthy', 'Northern Leaf Blight']
    
    test_images = tf.image.decode_jpeg(_image, channels=3)
    test_images = tf.image.resize(test_images, IMAGE_RES)
    # test_images = preprocessing.image.img_to_array(test_images)
    # test_images = test_images / 255.0
    test_images = np.expand_dims(test_images, axis=0)
    # print(test_images)

    predictions = model.predict(test_images)
    scores = tf.nn.softmax(predictions[0])
    # print(scores)
    scores = scores.numpy()
    print(scores)
    results = {
        'Cercospora Leaf Spot (Gray Leaf Spot)': 0,
        'Common Rust': 0,
        'Healthy': 0,
        'Northern Leaf Blight': 0
    }

    results = f"{classes[np.argmax(scores)]} with a { (100 * np.max(scores)).round(2)} % confidence score."

    if classes[np.argmax(scores)] == 'Cercospora Leaf Spot (Gray Leaf Spot)':
        st.markdown("""
        ### Disease Facts
        Gray leaf spot is caused by the fungus Cercospora zeae-maydis.

        Epidemics of gray leaf spot have been observed in New York State in the Southern Tier and the Hudson River Valley. New hot spots of the disease have been reported in the Mohawk Valley and the Leatherstocking Region.

        Gray leaf spot is favored by wet humid weather as often found in valley microclimates. Additionally, it is favored in situations with reduced tillage and continuous corn.

        Airborne spores are spread locally and regionally from corn debris.

        ### Symptoms

        Symptoms of gray leaf spot are usually first noticed in the lower leaves.

        Initially, lesions of gray leaf spot begin as a small dot with a yellow halo.

        Lesions will elongate over time running parallel to the veins becoming pale brown to gray and rectangular in shape with blunt ends. These lesions can be described as having the appearance of a ???matchstick.???

        Lesions may eventually coalesce killing the leaves.

        Leaves appear grayish in color due to the presence of fungal spores.

        ### Management Strategies

        Management strategies for gray leaf spot include tillage, crop rotation and planting resistant hybrids.

        Fungicides may be needed to prevent significant loss when plants are infected early and environmental conditions favor disease.
        """)

    if classes[np.argmax(scores)] == 'Common Rust':
        st.markdown("""
        ### Disease Facts
        Common rust is caused by the fungus Puccinia sorghi. Late occurring infections have limited impact on yield.

        The fungus overwinters on plants in southern states and airborne spores are wind-blown to northern states during the growing season.

        Disease development is favored by cool, moist weather (60 ??? 70??? F).

        ### Symptoms
        Symptoms of common rust often appear after silking.

        Small, round to elongate brown pustules form on both leaf surfaces and other above ground parts of the plant.

        As the pustules mature they become brown to black.

        If disease is severe, the leaves may yellow and die early.

        ## Management Strategies
        The use of resistant hybrids is the primary management strategy for the control of common rust.

        Timely planting of corn early during the growing season may help to avoid high inoculum levels or environmental conditions that would promote disease development.
        """)
    
    if classes[np.argmax(scores)] == 'Northern Leaf Blight':
        st.markdown("""
        ### Disease Facts
        Northern corn leaf blight caused by the fungus Exerohilum turcicum is a common leaf blight found in New York.

        If lesions begin early (before silking), crop loss can result. Late infections may have less of an impact on yield.

        Northern corn leaf blight is favored by wet humid cool weather typically found later in the growing season.

        Spores of the fungus that causes this disease can be transported by wind long distances from infected fields. Spread within and between fields locally also relies on wind blown spores.
        
        ### Symptoms
        The tan lesions of northern corn leaf blight are slender and oblong tapering at the ends ranging in size between 1 to 6 inches.

        Lesions run parallel to the leaf margins beginning on the lower leaves and moving up the plant. They may coalesce and cover the enter leaf.

        Spores are produced on the underside of the leaf below the lesions giving the appearance of a dusty green fuzz.

        ### Management Strategies
        Northern corn leaf blight can be managed through the use of resistant hybrids.

        Additionally, timely planting can be useful for avoiding conditions that favor the disease.
        """)
    
    return results


def main():
    st.sidebar.info('Survey Form: https://forms.gle/HFXHdhTN2K9oSnSM6')
    nav_select = st.sidebar.selectbox('Please select from the following', ['Classifier', 'Model Observation'])

    if nav_select == 'Model Observation':
        st.sidebar.success('Information are now shown on the right for desktop users.')
        st.title('Model Observation')

        st.header('Dataset')
        st.write("""
        The dataset used in this application is widely accesible on Kaggle 
        at (https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)""")

        st.header('Data Preprocessing')
        st.image('assets/Original Images Preview.png', caption='Before Data Augmentation')
        st.image('assets/Augmented Images Preview.png', caption='After Data Augmentation')
        st.image('assets/Distribution of Data.png', caption='Distribution of corn plant diseases training data')

        st.header('Accuracy/Loss Evolution Diagram')
        col1, col2 = st.columns(2)
        col1.image('assets/B0 Model Accuracy Plot History.png', caption='Accuracy Evolution Plot')
        col2.image('assets/B0 Model Loss Plot History.png', caption='Loss Evolution Plot')

        st.header('Actual v. Predicted Diagram')
        st.image('assets/Actual vs Predicted Diagram.png')

        st.header('Confusion Matrix')
        st.image('assets/Confusion Matrix Model B0.png')

        st.header('F1 Score')
        st.json("""
{"class_names":{"0":"Cercospora_leaf_spot Gray_leaf_spot","1":"Common_rust","2":"Healthy","3":"Northern_leaf_blight"},"f1_score":{"0":0.9493670886,"1":0.9846153846,"2":1.0,"3":0.9644670051}}
        """)


    if nav_select == 'Classifier':
        st.image('assets/corn-plant-diseases-list.png')
        st.title('Corn Plant Diseases Classifier')
        st.sidebar.success('You can now start classifying!')
        st.markdown(body)
        st.warning('It will take some time to classify, so please be patient.')
        # file_upload = st.file_uploader('Select an Image', type=['jpg', 'jpeg', 'png', 'gif'], accept_multiple_files=True)
        # submit_button = st.button('Classify')

        url_path = st.text_input('Enter Image URL To Classify...', value='https://static2.bigstockphoto.com/8/6/1/large2/168470984.jpg')
        submit_button = st.button('Classify')
        if submit_button:
            content = requests.get(url_path).content
            image = Image.open(BytesIO(content))
            st.image(image, use_column_width=True)
            predictions = predict(content)
            time.sleep(1)
            st.success('Image Has Been Classified')
            st.write(predictions)
        predict.clear()

        # if file_upload is not None:
        #     for img in file_upload:
        #         image = Image.open(img)
        #         st.markdown('----')
        #         st.image(image, caption='Image Preview', use_column_width=True)
        #         if submit_button:
        #             with st.spinner('Generating Informative Descriptions...'):
        #                 plt.imshow(image)
        #                 plt.axis('off')
        #                 predictions = predict(image)
        #                 time.sleep(1)
        #             st.success('Image Has Been Classified')
        #             st.write(predictions)
        #         predict.clear()

if __name__ == '__main__':
    main()