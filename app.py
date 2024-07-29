import os
import time
import pickle
import random
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l2
from streamlit_option_menu import option_menu
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

# Function to load images from a pickled file
@st.cache_data
def load_pickled_images(file_path):
    with open(file_path, 'rb') as f:
        images = pickle.load(f)
    return images

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image)
    image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=2)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_label = class_names[predicted_class[0]]
    return predicted_label, predictions[0][predicted_class[0]]

# Function to move misclassified image to correct folder
def move_image_to_correct_folder(image, correct_class):
    image_path = os.path.join('RealWaste2/train', correct_class, 'corrected_image.jpg')
    image.save(image_path)
    st.success(f"Image moved to {correct_class} folder")

# Function to retrain the model
def retrain_model():
    train_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    val_generator = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
    )

    train_data = train_generator.flow_from_directory(
        'RealWaste2/train',
        target_size=(224, 224),  
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    val_data = val_generator.flow_from_directory(
        'RealWaste2/val',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
        seed=0
    )

    y_train = train_data.classes

    cls_wt = class_weight.compute_class_weight(
        class_weight='balanced', 
        classes=np.unique(y_train), 
        y=y_train
    )

    class_weights = {i: cls_wt[i] for i in range(len(cls_wt))}

    pretrained_model = tf.keras.applications.ResNet50(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    pretrained_model.trainable = False
    inputs = pretrained_model.input

    x = tf.keras.layers.Dense(180, activation='relu', kernel_regularizer=l2(0.001))(pretrained_model.output)
    x = tf.keras.layers.Dense(360, activation='relu', kernel_regularizer=l2(0.001))(x)
    outputs = tf.keras.layers.Dense(8, activation='softmax')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    checkpoint = ModelCheckpoint("resnet50.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=2, min_delta=0.0001, verbose=1)

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=12,
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr]
    )

    with open('Pickle_files/models/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    st.success("Model retrained successfully")

# Load the pre-trained model from the .pkl file
@st.cache_resource
def load_model():
    with open('Pickle_files/models/resnet50.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Class names
class_names = ['Food Organics','Glass','Metal','Miscellaneous Trash','Paper','Plastic','Textile Trash','Vegetation']

# Company Logo
st.sidebar.image('Images/image4.png', use_column_width=True)

# Sidebar Selectbox initialization
selected = st.sidebar.selectbox("Go to", ["Home", "About", "Contact"])

# Path to your pickled images folder
pickled_images_folder = 'Pickle_files/images'

# Main Home Classification Page
if selected == "Home":

    # Function to randomly select one image from a list
    def select_one_image(images):
        return random.choice(images)
    
    # Home Page Option Menu Initialization
    selected1 = option_menu(
        menu_title=None,  
        options=["Home", "Material Handling", "Developer Mode"],  
        icons=["house", "file-earmark", "tools"],  
        menu_icon="cast",  
        default_index=0,  
        orientation="horizontal",
    )
    # Home Classification Page
    if selected1 == "Home":
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

    elif selected1 == "Material Handling":
        st.title("Material Handling Guidelines:")

        st.header("Compost Procedures", divider=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image('Images/Compost.jpg', width=120)
        with col2:
            st.write('Divertible Organics:')
            st.write('From which energy and fertilizer can be derived')
        st.write("Instructions to be given by customer")

        st.header("Recycle Procedures", divider=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image('Images/Recycle.jpg', width=120)
        with col2:
            st.write("Recyclable Inorganics:")
            st.write("Fit for repurposing")
        st.write("Instructions to be given by customer")

        st.header("Trash Procedures", divider=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image('Images/Trash.jpg', width=120)
        with col2:
            st.write("Inorganic Materials:")
            st.write("Requiring Landfill")
        st.write("Instructions to be given by customer")

    elif selected1 == "Developer Mode":
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

elif selected == "About":
    tab1, tab2 = st.tabs(["Software Used", "Hardware Used"])

    with tab1:
        st.header("Software Description", divider=True)
        st.write("COMING SOON......")

    with tab2:
        st.header("Hardware Description", divider=True)
        st.write("COMING SOON......")

elif selected == "Contact":
    st.header("DEVELOPER CONTACTS", divider= "rainbow")
    with st.container(border= True):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("Annabellah Mbungu   \nannbellah.mbungu@student.moringaschool.com     \nNumber:  ")  
        with col2:
            st.write("Brian Muthama  \nbrian.muthama@student.moringaschool.com   \nNumber:  ")
        with col3:
            st.write("Harris Lukundi  \nharris.lukundi@student.moringaschool.com  \nNumber:  ")
