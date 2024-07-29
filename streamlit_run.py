import streamlit as st

# Company Logo
st.sidebar.image('Images/image4.png', use_column_width=True)

# Sidebar Selectbox initialization
selected = st.sidebar.selectbox("Go to", ["Home", "About", "Contact"])

if selected == "Home":
    with open('main_page.py', 'r') as file:
        code = file.read()
        exec(code)