import streamlit as st

# Main Title
st.title("Material Handling Guidelines:")

# Section - Compost
st.header("Compost Procedures", divider=True)
col1, col2 = st.columns([3, 7])
with col1:
    st.image('Images/Compost.jpg', width=120)
    st.write("Instructions to be given by customer")
with col2:
    st.write('Divertible Organics:')
    st.write('Energy and fertilizers can be derived from this material.')
    with st.expander("Give Explicit instructions"):
        compost_instructions = st.text_area(
            label="Your compost instructions will be forwarded to our Team Lead Harris.",
            placeholder="Enter compost instructions here")
        if st.button("Submit", key="compost"):
            st.write(compost_instructions)

# Section - Recycle
st.header("Recycle Procedures", divider=True)
col1, col2 = st.columns([3, 7])
with col1:
    st.image('Images/Recycle.jpg', width=120)    
    st.write("Instructions to be given by customer")
with col2:
    st.write("Recyclable Inorganics:")
    st.write("Fit for repurposing")
    with st.expander("Give Explicit instructions"):
        recycle_instructions = st.text_area(
            label="Your recycle instructions will be forwarded to our Team Lead Harris.",
            placeholder="Enter recycle instructions here")
        if st.button("Submit", key="recycle"):
            st.write(recycle_instructions)

# Section - Trash
st.header("Trash Procedures", divider=True)
col1, col2 = st.columns([3, 7])
with col1:
    st.image('Images/Trash.jpg', width=120)
    st.write("Instructions to be given by customer")
with col2:
    st.write("Inorganic Materials:")
    st.write("Requiring Landfill")
    with st.expander("Give Explicit instructions"):
        trash_instructions = st.text_input(
            label="Your trash instructions will be forwarded to our Team Lead Harris.",
            placeholder="Enter trash instructions here")
        if st.button("Submit", key="trash"):
            st.write(trash_instructions)