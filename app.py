
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

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# AI TRAJECTORY PREDICTION SECTION
st.markdown("---")
st.subheader("🧠 AI Trajectory Prediction")

# Simple AI model training - FIXED
@st.cache_resource
def train_simple_ai_model():
    # Create simple training data
    np.random.seed(42)
    n_samples = 1000
    
    velocities = np.random.uniform(1, 100, n_samples)
    angles = np.random.uniform(1, 90, n_samples)
    
    # Calculate actual ranges using physics
    ranges = []
    for v, angle in zip(velocities, angles):
        angle_rad = np.radians(angle)
        range_val = (v**2 * np.sin(2 * angle_rad)) / 9.8
        ranges.append(range_val)
    
    # Create DataFrame
    df = pd.DataFrame({
        'velocity': velocities,
        'angle': angles,
        'range': ranges
    })
    
    # Train model
    X = df[['velocity', 'angle']]
    y = df['range']
    
    model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X, y)
    
    return model

# Load AI model
try:
    ai_model = train_simple_ai_model()
    st.success("✅ AI Model Trained Successfully!")
    
    # User input for AI prediction
    st.write("### Test AI Prediction")
    col1, col2 = st.columns(2)

    with col1:
        ai_velocity = st.slider("Velocity for AI (m/s)", 1, 100, 50, key="ai_vel")

    with col2:
        ai_angle = st.slider("Angle for AI (degrees)", 1, 90, 45, key="ai_ang")

    # Get AI prediction
    if st.button("Get AI Prediction"):
        ai_prediction = ai_model.predict([[ai_velocity, ai_angle]])[0]
        
        # Calculate actual physics result
        angle_rad = np.radians(ai_angle)
        actual_range = (ai_velocity**2 * np.sin(2 * angle_rad)) / 9.8
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🤖 AI Prediction", f"{ai_prediction:.1f} m")
            
        with col2:
            st.metric("📐 Physics Calculation", f"{actual_range:.1f} m")
            
        with col3:
            difference = abs(ai_prediction - actual_range)
            st.metric("📊 Difference", f"{difference:.2f} m")
        
        # Show accuracy
        accuracy_percent = max(0, 100 - (difference / actual_range * 100))
        st.progress(accuracy_percent / 100)
        st.write(f"**AI Accuracy: {accuracy_percent:.1f}%**")
        
        # Simple visualization
        st.write("### 📈 Comparison")
        fig, ax = plt.subplots(figsize=(8, 4))
        methods = ['AI Prediction', 'Physics Calculation']
        values = [ai_prediction, actual_range]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax.bar(methods, values, color=colors, alpha=0.7)
        ax.set_ylabel('Distance (meters)')
        ax.set_title('AI vs Physics Calculation')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                   f'{value:.1f}m', ha='center', va='bottom', fontweight='bold')
        
        st.pyplot(fig)
        
except Exception as e:
    st.error(f"❌ AI Model Error: {str(e)}")
    st.write("Trying alternative approach...")
    
    # Fallback: Simple linear approximation
    st.write("### 🔧 Using Simple AI Approximation")
    ai_velocity = st.slider("Velocity (m/s)", 1, 100, 50, key="simple_vel")
    ai_angle = st.slider("Angle (degrees)", 1, 90, 45, key="simple_ang")
    
    # Simple AI-like approximation
    simple_ai_pred = (ai_velocity ** 2 * ai_angle) / 100  # Simplified formula
    actual_range = (ai_velocity**2 * np.sin(2 * np.radians(ai_angle))) / 9.8
    
    st.metric("Simple AI Estimate", f"{simple_ai_pred:.1f} m")
    st.metric("Actual Physics", f"{actual_range:.1f} m")

# Educational content
st.write("---")
st.write("### 🧠 How This AI Works")
st.write("""
**Machine Learning Approach:**
1. **Training Data**: 1,000+ simulated projectile scenarios
2. **Algorithm**: Random Forest Regressor (ensemble learning)
3. **Features**: Velocity + Angle → Predicts Distance
4. **Learning**: Maps input parameters to output range

**Real-world Applications:**
- Sports analytics (projectile prediction)
- Military targeting systems
- Space mission trajectory planning
- Robotics and autonomous systems
""")
