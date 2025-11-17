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


 REAL-TIME SPRING ANIMATION
st.markdown("---")
st.subheader("Real-Time Spring Animation")

mass = st.slider("Mass (kg)", 0.1, 10.0, 1.0, key="mass_spring")
spring_constant = st.slider("Spring Constant (N/m)", 1, 100, 10, key="spring_const")

# Calculate physics
omega = np.sqrt(spring_constant / mass)
period = 2 * np.pi / omega

st.write(f"Oscillation Period: {period:.2f} seconds")

# Real-time animation

if st.button("Start Spring Animation"):
    spring_placeholder = st.empty()
    progress_placeholder = st.empty()
    text_placeholder = st.empty()
    
    start_time = time.time()
    
    while True:
        # Calculate current time in oscillation
        current_time = time.time() - start_time
        position = np.sin(omega * current_time)
        
        # Convert to progress bar value (0 to 1)
        progress_val = (position + 1) / 2
        
        # Update animation
        spring_placeholder.markdown(f"""
        <div style="text-align: center; font-size: 20px;">
        🟦<br>
        {'│' * int(10 + position * 5)}<br>
        ⚫ Mass<br>
        Position: {position:.2f}
        </div>
        """, unsafe_allow_html=True)
        
        progress_placeholder.progress(progress_val)
        text_placeholder.write(f"Time: {current_time:.1f}s | Position: {position:.2f}")
        
        # Small delay
        time.sleep(0.1)
        
        # Stop after 10 seconds or if user refreshes
        if current_time > 10:
            break
