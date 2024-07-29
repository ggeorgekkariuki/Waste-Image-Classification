import streamlit as st
from classes import cool_excuse

# Company Logo
st.sidebar.image('Images/image4.png', use_column_width=True)

# Sidebar Selectbox initialization
selected = st.sidebar.selectbox("Go to", ["Home", "About", "Contact"])

if selected == "Home":
    with open('main_page.py', 'r') as file:
        code = file.read()
        exec(code)

elif selected == "About":
    tab1, tab2 = st.tabs(["Software Used", "Hardware Used"])
    
    with tab1:
        st.header("Software Description", divider=True)
        st.write(f"{cool_excuse()} Page Coming Soon...")

    with tab2:
        st.header("Hardware Description", divider=True)
        st.write(f"{cool_excuse()} Page Coming Soon...")

elif selected == "Contact":
    st.header("Developer Contacts", divider= "rainbow")
    with st.container(border= True):
        col1, col2 = st.columns(2)
        with col2:
            st.write("""
                     George Kariuki :goggles:  \nTeam Lead Assistant :hatching_chick:
                     george.kariuki1@student.moringaschool.com  \nNumber:*Unavailable*""")
        with col1:
            st.write("""Harris Lukundi :sunglasses:  \nTeam Lead :crown:
                     harris.lukundi@student.moringaschool.com  \nNumber:*Unavailable*  """)