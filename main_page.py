import streamlit as st
from streamlit_option_menu import option_menu

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
    with open('home_page.py', 'r') as file:
        code = file.read()
        exec(code)