from classes import *
import streamlit as st

with st.container(border=True):
    st.image('Images/image1.webp', use_column_width=True)
    st.write("<h1 style='text-align: center;'>IDump Classification System</h1>", unsafe_allow_html=True)

    with st.sidebar:
        st.header('Landfill Samples')

        if os.path.isdir(pickled_images_folder):
            pickled_files = [f for f in os.listdir(pickled_images_folder) if f.lower().endswith('.pkl')]

            if pickled_files:
                selected_images = []

                for pickled_file in pickled_files:
                    images = load_pickled_images(os.path.join(pickled_images_folder, pickled_file))
                    image = select_one_image(images)
                    selected_images.append(image)

                num_columns = 2
                cols = st.sidebar.columns(num_columns)
                for idx, (image, filename) in enumerate(zip(selected_images, class_names)):
                    col = cols[idx % num_columns]
                    image = image.replace("\\", "/")
                    col.image(image, caption=filename, use_column_width=True)
            else:
                st.sidebar.warning('No pickled files found in the specified folder.')
        else:
            st.sidebar.error(f'The directory "{pickled_images_folder}" does not exist. Please check the path and try again.')

    
    uploaded_files = st.file_uploader("Choose images", type=['jpg', 'png', 'jpeg'], accept_multiple_files=True)
        #uploaded_files = st.camera_input("Choose images")

    image_details = []

    if 'last_uploaded_image' not in st.session_state:
        st.session_state.last_uploaded_image = None

    col1, col2 = st.columns(2, gap="small")
    
    with col1:
        with st.container(border=True):
            if uploaded_files:
                image = Image.open(uploaded_files[-1])  
                st.session_state.last_uploaded_image = image
                
                st.image(image, width=280)
                
                image_details.append({  
                    'Filename': uploaded_files[-1].name,
                    'Width': image.width,
                    'Height': image.height
                })
                
                if image_details:
                    df = pd.DataFrame(image_details)
                    df.set_index("Filename", inplace=True)
                    st.write(df)

    with col2:
        with st.container(border=True):
            if st.session_state.last_uploaded_image:
                with st.spinner('Classifying...'):
                    time.sleep(2)  
                
                st.success('Done!')
                
                try:
                    label, confidence = predict(st.session_state.last_uploaded_image)
                    
                    st.write(f"PREDICTED CLASS: {label}")
                    st.write(f"CONFIDENCE: {confidence * 100:.2f}%")

                    st.subheader('MATERIAL HANDLING', divider=True)
                    if label == 'Food Organics' or label == 'Vegetation':
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image('Images/Compost.jpg', width=120)
                        with col2:
                            st.write('Divertible Organics: From which energy and fertilizer can be derived')
                    elif label in ['Glass', 'Paper', 'Metal', 'Plastic']:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image('Images/Recycle.jpg', width=120)
                        with col2:
                            st.write("Recyclable Inorganics: Fit for repurposing")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image('Images/Trash.jpg', width=120)
                        with col2:
                            st.write("Inorganic Materials: Requiring Landfill ")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

    with st.container(border=True):    
        if st.session_state.last_uploaded_image:
            st.write("Was the Image Classified Correctly?")
            choice = st.radio("", ("Select an Option","👍 Yes", "👎 No"), index=0)
            if choice == "👍 Yes":
                st.success("Classification correct")
            elif choice == "👎 No":
                st.session_state.wrongly_classified_image = st.session_state.last_uploaded_image
                st.warning("Misclassified image has been moved to 'Developer Mode' for further action.")