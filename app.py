
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# AI TRAJECTORY PREDICTION SECTION
st.markdown("---")
st.subheader("🧠 AI Trajectory Prediction")

# Generate training data for the AI model
@st.cache_data
def generate_training_data():
    data = []
    for v in range(1, 101):  # Velocity from 1-100 m/s
        for angle in range(1, 91):  # Angle from 1-90 degrees
            # Physics calculations for training data
            angle_rad = np.radians(angle)
            time_flight = (2 * v * np.sin(angle_rad)) / 9.8
            max_height = (v * np.sin(angle_rad))**2 / (2 * 9.8)
            range_val = (v**2 * np.sin(2 * angle_rad)) / 9.8
            
            data.append([v, angle, time_flight, max_height, range_val])
    
    return pd.DataFrame(data, columns=['velocity', 'angle', 'time_flight', 'max_height', 'range'])

# Train the AI model
@st.cache_resource
def train_ai_model():
    df = generate_training_data()
    
    # Features: velocity and angle
    X = df[['velocity', 'angle']]
    # Target: range (distance)
    y = df['range']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = mean_absolute_error(y_test, y_pred)
    
    return model, accuracy

# Load AI model
model, model_accuracy = train_ai_model()

st.write(f"**AI Model Accuracy:** ±{model_accuracy:.2f} meters")

# User input for AI prediction
st.write("### Test AI Prediction")
col1, col2 = st.columns(2)

with col1:
    ai_velocity = st.slider("Velocity for AI (m/s)", 1, 100, 50, key="ai_velocity")

with col2:
    ai_angle = st.slider("Angle for AI (degrees)", 1, 90, 45, key="ai_angle")

# Get AI prediction
ai_prediction = model.predict([[ai_velocity, ai_angle]])[0]

# Calculate actual physics result for comparison
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
    st.metric("📊 Difference", f"{difference:.1f} m")

# Visual comparison
st.write("### 📈 AI vs Physics Comparison")
comparison_data = pd.DataFrame({
    'Method': ['AI Prediction', 'Physics Calculation'],
    'Distance (m)': [ai_prediction, actual_range]
})

fig, ax = plt.subplots()
bars = ax.bar(comparison_data['Method'], comparison_data['Distance (m)'], color=['#FF6B6B', '#4ECDC4'])
ax.set_ylabel('Distance (meters)')
ax.set_title('AI vs Traditional Physics Calculation')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}m', ha='center', va='bottom')

st.pyplot(fig)

# Explanation
st.write("### 🔍 How the AI Works")
st.write("""
- **Training Data**: The AI was trained on 9,000+ projectile motion scenarios
- **Algorithm**: Uses Random Forest Regressor (ensemble machine learning)
- **Learning**: Learned the relationship between velocity, angle, and resulting distance
- **Purpose**: Demonstrates how AI can approximate complex physics calculations
- **Accuracy**: Shows the difference between AI prediction and exact physics
""")
