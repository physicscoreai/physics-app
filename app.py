import streamlit as st

st.title("Physics Simulator")
st.write("Welcome to my physics app!")

velocity = st.slider("Choose Velocity", 1, 100, 50)
angle = st.slider("Choose Angle", 1, 90, 45)

st.write("You selected velocity:", velocity)
st.write("You selected angle:", angle)
