import streamlit as st
from streamlit_option_menu import option_menu

# Home Page Option Menu Initialization
selected_option = option_menu(
    menu_title=None,  
    options=["Home", "Material Handling", "Developer Mode"],  
    icons=["house", "file-earmark", "tools"],  
    menu_icon="cast",  
    default_index=0,  
    orientation="horizontal",
)

# Home Classification Tab
if selected_option == "Home":
    with open('home_page.py', 'r') as file:
        code = file.read()
        exec(code)

# Material Handling Tab
elif selected_option == "Material Handling":
    with open('material_handling.py', 'r') as file:
        code = file.read()
        exec(code)    

# Developer Tab
elif selected_option == "Developer Mode":
    with open('developer_page.py', 'r') as file:
        code = file.read()
        exec(code)   