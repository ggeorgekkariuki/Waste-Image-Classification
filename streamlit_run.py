import streamlit as st

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