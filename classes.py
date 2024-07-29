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

# Function to randomly select one image from a list
def select_one_image(images):
    return random.choice(images)

# Path to your pickled images folder
pickled_images_folder = 'Pickle_files/images'

# Garbage Class names
class_names = ['Food Organics','Glass','Metal','Miscellaneous Trash','Paper','Plastic','Textile Trash','Vegetation']

# Cool excuses
excuses = ["Oh no! The horror! ", "I swear it was here yesterday!", 
           "The dog age my homework! Walahi Walahi boss!", "Please avert your eyes! The shame!", 
           "It was the intern's fault!", "Reject Finance Bill"]

def cool_excuse():
    return np.random.choice(excuses, 1)[0]