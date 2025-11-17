st.title("Physics Simulator")

velocity = st.slider("Initial Velocity (m/s)", 1, 100, 50)
angle = st.slider("Launch Angle (degrees)", 1, 90, 45)

# Simple calculation without matplotlib
angle_rad = np.radians(angle)
g = 9.8
time = 2 * velocity * np.sin(angle_rad) / g
max_height = (velocity * np.sin(angle_rad))**2 / (2 * g)
distance = velocity * np.cos(angle_rad) * time

st.write(f"Time of flight: {time:.2f} seconds")
st.write(f"Max height: {max_height:.2f} meters") 
st.write(f"Distance: {distance:.2f} meters")
