import streamlit as st
import numpy as np

st.title("Physics Simulator - Projectile Motion")

# Input sliders
velocity = st.slider("Initial Velocity (m/s)", 1, 100, 50)
angle = st.slider("Launch Angle (degrees)", 1, 90, 45)

# Physics calculations
angle_rad = np.radians(angle)
g = 9.8
time_of_flight = 2 * velocity * np.sin(angle_rad) / g
max_height = (velocity * np.sin(angle_rad))**2 / (2 * g)
distance = velocity * np.cos(angle_rad) * time_of_flight

# Display results
st.subheader("Physics Results:")
st.write(f"Time of flight: {time_of_flight:.2f} seconds")
st.write(f"Maximum height: {max_height:.2f} meters")
st.write(f"Total distance: {distance:.2f} meters")

# Simple visual representation
st.subheader("Trajectory Preview:")
st.progress(angle/90)  # Shows angle as progress bar
st.metric("Power", f"{velocity} m/s")
