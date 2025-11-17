import streamlit as st
import numpy as np

st.title("Physics Simulator - Projectile Motion")

# Input sliders
velocity = st.slider("Initial Velocity (m/s)", 1, 100, 50)
angle = st.slider("Launch Angle (degrees)", 1, 90, 45)

# Physics Equations
st.subheader("Physics Equations Used:")
st.latex(r"x(t) = v_0 \cdot \cos(\theta) \cdot t")
st.latex(r"y(t) = v_0 \cdot \sin(\theta) \cdot t - \frac{1}{2} g t^2")
st.latex(r"T = \frac{2 v_0 \sin(\theta)}{g}")
st.latex(r"H = \frac{(v_0 \sin(\theta))^2}{2g}")
st.latex(r"R = \frac{v_0^2 \sin(2\theta)}{g}")

# Physics calculations
angle_rad = np.radians(angle)
g = 9.8
time = 2 * velocity * np.sin(angle_rad) / g

# Calculate trajectory points
t = np.linspace(0, time, 50)
x = velocity * np.cos(angle_rad) * t
y = velocity * np.sin(angle_rad) * t - 0.5 * g * t**2

# Display the chart
st.subheader("Projectile Trajectory")
st.line_chart(y)

# Show results
st.subheader("Physics Results:")
st.write(f"Time of flight: {time:.2f} seconds")
st.write(f"Maximum height: {max(y):.2f} meters") 
st.write(f"Total distance: {max(x):.2f} meters")


# NEW SIMULATION: Spring Mass
st.markdown("---")
st.subheader("Spring Mass Simulation")

mass = st.slider("Mass (kg)", 0.1, 10.0, 1.0)
spring_constant = st.slider("Spring Constant (N/m)", 1, 100, 10)

# Spring physics
omega = np.sqrt(spring_constant / mass)
period = 2 * np.pi / omega

st.write(f"Oscillation Period: {period:.2f} seconds")
st.write(f"Angular Frequency: {omega:.2f} rad/s")

# Show spring compression/extension
st.subheader("Spring Motion")
st.progress(0.5)
st.write("Spring Position")

# SPRING VISUALIZATION
st.markdown("---")
st.subheader("Spring Visualization")

mass = st.slider("Mass (kg)", 0.1, 10.0, 1.0, key="mass_spring")
spring_constant = st.slider("Spring Constant (N/m)", 1, 100, 10, key="spring_const")
displacement = st.slider("Spring Displacement", -1.0, 1.0, 0.0, 0.1)

# Calculate force
force = -spring_constant * displacement

# Draw spring based on displacement
st.write("**Spring Position:**")

if displacement > 0:
    # Spring stretched down
    st.code(f"""
    🟦 Wall
    {'│' * 8}
    ⚫ Mass (stretched down)
    Force: {force:.1f} N upward
    """)
elif displacement < 0:
    # Spring compressed up
    st.code(f"""
    🟦 Wall
    ⚫ Mass (compressed up)
    {'│' * 8}
    Force: {force:.1f} N downward
    """)
else:
    # Spring at rest
    st.code(f"""
    🟦 Wall
    {'│' * 8}
    ⚫ Mass (at rest)
    Force: {force:.1f} N
    """)

st.write(f"**Spring Force: {force:.1f} N**")
st.write(f"*Move the displacement slider to see the spring change!*")
