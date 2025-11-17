import streamlit as st
st.title("My Physics App")
st.write("Hello World!")
number = st.slider("Pick number", 1, 100)
st.write(f"You picked: {number}")
