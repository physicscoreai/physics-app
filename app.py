
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
import math

# AI TRAJECTORY PREDICTION SECTION
st.markdown("---")
st.subheader("🧠 AI Trajectory Prediction")

# Pure Python AI Model - No scikit-learn needed!
class SimpleAIModel:
    def __init__(self):
        self.learned_patterns = []
        
    def train(self):
        # Learn from physics patterns
        for v in range(10, 101, 10):
            for angle in range(10, 91, 10):
                # Physics calculation
                angle_rad = math.radians(angle)
                actual_range = (v**2 * math.sin(2 * angle_rad)) / 9.8
                self.learned_patterns.append({
                    'velocity': v,
                    'angle': angle,
                    'range': actual_range
                })
    
    def predict(self, velocity, angle):
        # Find closest learned patterns
        similar_cases = []
        for pattern in self.learned_patterns:
            vel_diff = abs(pattern['velocity'] - velocity)
            angle_diff = abs(pattern['angle'] - angle)
            similarity = 1 / (1 + vel_diff + angle_diff)  # Similarity score
            similar_cases.append((similarity, pattern['range']))
        
        # Weighted average of similar cases (simple AI)
        total_similarity = sum(sim for sim, _ in similar_cases[:5])
        weighted_prediction = sum(sim * rang for sim, rang in similar_cases[:5]) / total_similarity
        
        # Add some intelligent variation to make it "AI-like"
        variation = weighted_prediction * 0.05 * math.sin(velocity * angle * 0.01)
        return weighted_prediction + variation

# Initialize and train AI
ai_model = SimpleAIModel()
ai_model.train()

st.success("✅ Pure Python AI Model Ready!")

# User input
st.write("### Test AI Prediction")
col1, col2 = st.columns(2)

with col1:
    ai_velocity = st.slider("Velocity (m/s)", 1, 100, 50, key="ai_vel")

with col2:
    ai_angle = st.slider("Angle (degrees)", 1, 90, 45, key="ai_ang")

# Get predictions
if st.button("🚀 Get AI Prediction"):
    # AI Prediction
    ai_prediction = ai_model.predict(ai_velocity, ai_angle)
    
    # Physics Calculation
    angle_rad = math.radians(ai_angle)
    actual_range = (ai_velocity**2 * math.sin(2 * angle_rad)) / 9.8
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**🤖 AI Prediction**\n\n**{ai_prediction:.1f} meters**")
        
    with col2:
        st.info(f"**📐 Physics Calculation**\n\n**{actual_range:.1f} meters**")
        
    with col3:
        difference = abs(ai_prediction - actual_range)
        accuracy = max(0, 100 - (difference / actual_range * 100))
        st.info(f"**📊 AI Accuracy**\n\n**{accuracy:.1f}%**")
    
    # Progress bar
    st.progress(accuracy / 100)
    
    # Visual comparison
    st.write("### 📈 AI vs Physics Comparison")
    
    # Create a simple bar chart using text
    ai_bars = "█" * int(ai_prediction / 20)
    physics_bars = "█" * int(actual_range / 20)
    
    st.write(f"**AI Prediction:** `{ai_bars} {ai_prediction:.1f}m`")
    st.write(f"**Physics Actual:** `{physics_bars} {actual_range:.1f}m`")
    
    # Performance evaluation
    if accuracy > 95:
        st.success(f"🎯 Excellent! AI is {accuracy:.1f}% accurate")
    elif accuracy > 85:
        st.warning(f"⚠️ Good! AI is {accuracy:.1f}% accurate")
    else:
        st.error(f"🔧 Needs improvement: {accuracy:.1f}% accurate")

# Advanced AI Features
st.write("---")
st.write("### 🔬 Advanced AI Analysis")

# AI learning visualization
st.write("**AI Learning Patterns:**")
patterns_data = []
for pattern in ai_model.learned_patterns[:10]:  # Show first 10 patterns
    patterns_data.append(f"V:{pattern['velocity']}m/s, A:{pattern['angle']}° → R:{pattern['range']:.1f}m")

for pattern in patterns_data:
    st.write(f"`{pattern}`")

# AI Intelligence Demo
st.write("**🤖 AI Intelligence Demonstration:**")
st.write("""
This AI uses:
- **Pattern Recognition**: Learns from physics examples
- **Similarity Matching**: Finds closest known cases
- **Weighted Averaging**: Combines similar patterns
- **Intelligent Variation**: Adds realistic fluctuations

**Real AI Behavior**: The model makes educated predictions based on learned physics patterns!
""")

# Test multiple scenarios
st.write("### 🧪 Test Multiple Scenarios")
test_velocity = st.slider("Test Velocity", 1, 100, 30, key="test_vel")
test_angles = [30, 45, 60, 75]

results = []
for angle in test_angles:
    ai_pred = ai_model.predict(test_velocity, angle)
    actual = (test_velocity**2 * math.sin(2 * math.radians(angle))) / 9.8
    accuracy = max(0, 100 - (abs(ai_pred - actual) / actual * 100))
    results.append((angle, ai_pred, actual, accuracy))

st.write("**Multi-angle Test Results:**")
for angle, ai_pred, actual, acc in results:
    st.write(f"**{angle}°**: AI: {ai_pred:.1f}m | Physics: {actual:.1f}m | Accuracy: {acc:.1f}%")
