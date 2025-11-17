import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

st.title("Physics Simulator")

velocity = st.slider("Initial Velocity (m/s)", 1, 100, 50)
angle = st.slider("Launch Angle (degrees)", 1, 90, 45)

angle_rad = np.radians(angle)
g = 9.8
time = 2 * velocity * np.sin(angle_rad) / g
t = np.linspace(0, time, 100)
x = velocity * np.cos(angle_rad) * t
y = velocity * np.sin(angle_rad) * t - 0.5 * g * t**2

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("Distance (m)")
ax.set_ylabel("Height (m)")
ax.set_title("Projectile Motion")
st.pyplot(fig)
