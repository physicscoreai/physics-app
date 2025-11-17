
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

# AI TRAJECTORY PREDICTION SECTION
st.markdown("---")
st.subheader("🧠 AI Trajectory Prediction")

# Simple AI model training - No matplotlib required
@st.cache_resource
def train_simple_ai_model():
    # Create training data
    np.random.seed(42)
    n_samples = 500
    
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
    
    # Train simple model
    X = df[['velocity', 'angle']]
    y = df['range']
    
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=30, random_state=42, max_depth=8)
    model.fit(X, y)
    
    return model

try:
    # Load AI model
    ai_model = train_simple_ai_model()
    st.success("✅ AI Model Loaded Successfully!")
    
    # User input
    st.write("### Test AI Prediction")
    col1, col2 = st.columns(2)

    with col1:
        ai_velocity = st.slider("Velocity (m/s)", 1, 100, 50, key="ai_vel")

    with col2:
        ai_angle = st.slider("Angle (degrees)", 1, 90, 45, key="ai_ang")

    # Get predictions when button clicked
    if st.button("🚀 Get AI Prediction"):
        ai_prediction = ai_model.predict([[ai_velocity, ai_angle]])[0]
        
        # Calculate actual physics result
        angle_rad = np.radians(ai_angle)
        actual_range = (ai_velocity**2 * np.sin(2 * angle_rad)) / 9.8
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**🤖 AI Prediction**\n\n**{ai_prediction:.1f} meters**")
            
        with col2:
            st.info(f"**📐 Physics Calculation**\n\n**{actual_range:.1f} meters**")
            
        with col3:
            difference = abs(ai_prediction - actual_range)
            accuracy = max(0, 100 - (difference / actual_range * 100))
            st.info(f"**📊 AI Accuracy**\n\n**{accuracy:.1f}%**")
        
        # Visual progress bar
        st.progress(accuracy / 100)
        
        # Simple text-based visualization
        st.write("### 📈 Distance Comparison")
        st.write(f"`AI: {'█' * int(ai_prediction/10)} {ai_prediction:.1f}m`")
        st.write(f"`Physics: {'█' * int(actual_range/10)} {actual_range:.1f}m`")
        
        # Show difference
        if difference < 5:
            st.success(f"✅ AI is very accurate! Only {difference:.2f}m difference")
        elif difference < 15:
            st.warning(f"⚠️ AI is reasonably accurate: {difference:.2f}m difference")
        else:
            st.error(f"❌ AI needs improvement: {difference:.2f}m difference")

except Exception as e:
    st.error("❌ AI model failed. Using fallback calculation.")
    
    # Fallback: Simple demonstration
    st.write("### 🔧 AI Demonstration Mode")
    ai_velocity = st.slider("Velocity (m/s)", 1, 100, 50, key="fallback_vel")
    ai_angle = st.slider("Angle (degrees)", 1, 90, 45, key="fallback_ang")
    
    # Simple AI approximation
    simple_ai_pred = (ai_velocity ** 2 * np.sin(2 * np.radians(ai_angle))) / 9.5  # Slightly different constant
    
    # Actual physics
    actual_range = (ai_velocity**2 * np.sin(2 * np.radians(ai_angle))) / 9.8
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("AI Estimate", f"{simple_ai_pred:.1f} m")
    with col2:
        st.metric("Physics Actual", f"{actual_range:.1f} m")

# Educational section
st.write("---")
st.write("### 🧠 About This AI Model")
st.write("""
**Machine Learning Details:**
- **Algorithm**: Random Forest Regressor
- **Training Data**: 500+ projectile scenarios
- **Features**: Velocity + Launch Angle
- **Target**: Landing Distance

**How AI Learns Physics:**
- Studies patterns between inputs and outputs
- Builds decision trees to make predictions
- Combines multiple trees for better accuracy
- Can generalize to new, unseen scenarios
""")
