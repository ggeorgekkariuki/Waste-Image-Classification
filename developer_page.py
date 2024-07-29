import streamlit as st
from classes import *

st.title("Developer Mode")
with st.container(border=True):
    if 'wrongly_classified_image' in st.session_state and st.session_state.wrongly_classified_image:
        col1, col2= st.columns(2)
        with col1:
            image = st.session_state.wrongly_classified_image
            st.image(image, caption="Wrongly Classified Image", use_column_width=True)
            class_names = ['Food Organics','Glass','Metal','Miscellaneous Trash','Paper','Plastic','Textile Trash','Vegetation']
        with col2:
            correct_class = st.selectbox("Select the correct class:", class_names)
            if st.button("Confirm Correction"):
                if correct_class:
                    move_image_to_correct_folder(image, correct_class) 
                    if st.button("TEACH"):
                        st.write("Kindly Be Patient......")
                        st.write("This will take up to an hour!!!")
                        retrain_model()
    # elif 'wrongly_classified_image' not in st.session_state:
    #     st.info("Please redirect to the Home Page.")
    else:
        st.info("Please redirect to the Home Page.")