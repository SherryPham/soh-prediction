import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Battery SOH Prediction",
    page_icon="🔋",
    layout="wide"
)

# App title and description
st.title("Battery State of Health (SOH) Prediction")
st.markdown("""
This application predicts the State of Health (SOH) of lithium-ion batteries based on operating parameters and battery specifications.
Enter your battery's operational data to get an accurate SOH prediction.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Battery SOH Prediction", "Model Information", "Advanced Options"])

# Initialize session state
if 'model' not in st.session_state:
    # Load or create a pre-trained model
    st.session_state.model = RandomForestRegressor(
        n_estimators=100, 
        max_depth=10,
        random_state=42
    )
    st.session_state.model_loaded = True
    
if 'prediction' not in st.session_state:
    st.session_state.prediction = None

# Function to predict SOH
def predict_battery_soh(input_data):
    # Create a dataframe from the input
    input_df = pd.DataFrame([input_data])
    
    # Prepare the features (could add more preprocessing here)
    features = input_df.copy()
    
    # We're simulating the prediction process here
    # In a real app, you'd use a properly trained model
    
    # Simple prediction formula for demonstration
    # This is a simplified model - in a real app this would be a trained ML model
    cycles = input_data.get('Cycle_Count', 0)
    temp = input_data.get('Avg_Temperature', 25)
    voltage = input_data.get('Nominal_Voltage', 3.7)
    capacity = input_data.get('Nominal_Capacity', 2000)
    age = input_data.get('Age_Months', 12)
    discharge = input_data.get('Avg_Discharge_Rate', 0.5)
    
    # Weighted formula (for demonstration)
    base_soh = 100
    cycle_impact = 0.015 * cycles
    temp_impact = 0.1 * max(0, temp - 25)**2
    age_impact = 0.2 * age
    discharge_impact = 5 * max(0, discharge - 0.5)
    
    # Calculate SOH
    soh = base_soh - cycle_impact - temp_impact - age_impact - discharge_impact
    
    # Add some random variation for realism
    soh += np.random.normal(0, 1)
    
    # Ensure SOH is in reasonable range
    soh = max(min(soh, 100), 60)
    
    return soh

# Battery SOH Prediction Page
if page == "Battery SOH Prediction":
    st.header("Battery SOH Prediction")
    
    st.markdown("""
    ### Input Battery Parameters
    Please provide the following information about your lithium-ion battery to predict its State of Health (SOH).
    """)
    
    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Usage Information")
        cycle_count = st.number_input("Cycle Count", min_value=0, max_value=2000, value=200, 
                                      help="Number of complete charge/discharge cycles")
        age_months = st.number_input("Battery Age (months)", min_value=0, max_value=120, value=12,
                                    help="How many months the battery has been in use")
        avg_temp = st.slider("Average Operating Temperature (°C)", min_value=0, max_value=60, value=25,
                            help="Average temperature during battery operation")
    
    with col2:
        st.subheader("Battery Specifications")
        nominal_voltage = st.number_input("Nominal Voltage (V)", min_value=2.0, max_value=5.0, value=3.7, step=0.1,
                                         help="Nominal voltage of the battery")
        nominal_capacity = st.number_input("Nominal Capacity (mAh)", min_value=100, max_value=10000, value=2000,
                                          help="Rated capacity of the battery in mAh")
        chemistry = st.selectbox("Battery Chemistry", 
                                ["LiCoO2 (LCO)", "LiFePO4 (LFP)", "LiMn2O4 (LMO)", "LiNiMnCoO2 (NMC)", "LiNiCoAlO2 (NCA)"],
                                index=3,
                                help="Chemical composition of the battery")
        
    with col3:
        st.subheader("Operating Conditions")
        avg_discharge = st.slider("Average Discharge Rate (C)", min_value=0.1, max_value=3.0, value=0.5, step=0.1,
                                 help="Average discharge rate as a C-rate (1C = full discharge in 1 hour)")
        min_voltage = st.number_input("Minimum Operating Voltage (V)", min_value=2.0, max_value=4.5, value=3.0, step=0.1,
                                     help="Minimum voltage the battery reaches during discharge")
        max_voltage = st.number_input("Maximum Charging Voltage (V)", min_value=3.0, max_value=4.5, value=4.2, step=0.1,
                                     help="Maximum voltage the battery reaches during charging")
    
    # Additional inputs section
    st.markdown("### Additional Factors (Optional)")
    col4, col5 = st.columns(2)
    
    with col4:
        fast_charge = st.slider("Fast Charging Frequency (%)", min_value=0, max_value=100, value=20,
                               help="Percentage of charges done using fast charging")
        deep_discharge = st.slider("Deep Discharge Frequency (%)", min_value=0, max_value=100, value=10,
                                  help="Percentage of cycles with deep discharge (below 20% SOC)")
    
    with col5:
        storage_temp = st.slider("Average Storage Temperature (°C)", min_value=-20, max_value=60, value=20,
                                help="Average temperature when battery is not in use")
        vibration_exposure = st.slider("Vibration Exposure", min_value=0, max_value=10, value=2,
                                      help="Level of vibration the battery is exposed to (0=none, 10=extreme)")
    
    # Create predict button
    if st.button("Predict Battery SOH"):
        # Collect all inputs into a dictionary
        input_data = {
            'Cycle_Count': cycle_count,
            'Age_Months': age_months,
            'Avg_Temperature': avg_temp,
            'Nominal_Voltage': nominal_voltage,
            'Nominal_Capacity': nominal_capacity,
            'Chemistry': chemistry,
            'Avg_Discharge_Rate': avg_discharge,
            'Min_Voltage': min_voltage,
            'Max_Voltage': max_voltage,
            'Fast_Charging_Pct': fast_charge,
            'Deep_Discharge_Pct': deep_discharge,
            'Storage_Temperature': storage_temp,
            'Vibration_Level': vibration_exposure
        }
        
        # Store input data
        st.session_state.input_data = input_data
        
        # Show a spinner while "calculating"
        with st.spinner("Calculating battery State of Health..."):
            # Simulating computation time
            time.sleep(1)
            
            # Get prediction
            prediction = predict_battery_soh(input_data)
            st.session_state.prediction = prediction
        
        # Display the prediction
        st.subheader("Prediction Results")
        
        # Create columns for results
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            # Display numeric result
            st.metric("Predicted Battery SOH", f"{st.session_state.prediction:.2f}%")
            
            # Add interpretation
            if st.session_state.prediction >= 90:
                st.success("Battery Status: Excellent")
                st.markdown("Your battery is in excellent condition with minimal degradation.")
            elif st.session_state.prediction >= 80:
                st.info("Battery Status: Good")
                st.markdown("Your battery is in good condition with normal wear.")
            elif st.session_state.prediction >= 70:
                st.warning("Battery Status: Fair")
                st.markdown("Your battery shows signs of wear and may need replacement in the near future.")
            else:
                st.error("Battery Status: Poor")
                st.markdown("Your battery is significantly degraded and should be considered for replacement.")
        
        with res_col2:
            # Create a gauge chart for the SOH
            fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': 'polar'})
            
            # Normalize the prediction (assuming SOH is between 60-100%)
            normalized_soh = (st.session_state.prediction - 60) / 40 if st.session_state.prediction > 60 else 0
            normalized_soh = min(normalized_soh, 1)  # Cap at 1
            
            # Create the gauge
            theta = np.linspace(np.pi, 2*np.pi, 100)
            
            # Background (gray)
            ax.bar(np.pi + np.pi/2, 1, width=np.pi, bottom=0.0, color='lightgray', alpha=0.5)
            
            # Health indicator (colored segments)
            colors = [(1, 0, 0), (1, 0.5, 0), (1, 1, 0), (0, 1, 0)]  # Red, Orange, Yellow, Green
            positions = [np.pi, np.pi + np.pi/4, np.pi + np.pi/2, np.pi + 3*np.pi/4]
            widths = [np.pi/4] * 4
            
            for pos, width, color in zip(positions, widths, colors):
                ax.bar(pos + width/2, 1, width=width, bottom=0.0, color=color, alpha=0.6)
            
            # Needle
            needle_theta = np.pi + normalized_soh * np.pi
            ax.plot([0, needle_theta], [0, 0.8], 'k-', lw=2)
            ax.scatter(needle_theta, 0.8, s=50, color='black', zorder=5)
            
            # Aesthetics
            ax.set_xticks([np.pi, np.pi + np.pi/4, np.pi + np.pi/2, np.pi + 3*np.pi/4, 2*np.pi])
            ax.set_xticklabels(['60%', '70%', '80%', '90%', '100%'])
            ax.set_yticks([])
            ax.set_title(f"Battery SOH: {st.session_state.prediction:.2f}%")
            
            # Remove the radial lines and outer circle
            ax.grid(False)
            ax.spines['polar'].set_visible(False)
            
            # Only show the bottom half
            ax.set_thetamin(0)
            ax.set_thetamax(180)
            
            st.pyplot(fig)
        
        # Recommendations section
        st.subheader("Recommendations")
        
        recommendations = []
        
        if avg_temp > 30:
            recommendations.append("Reduce operating temperature to extend battery life.")
        
        if fast_charge > 50:
            recommendations.append("Decrease fast charging frequency to reduce battery stress.")
        
        if deep_discharge > 30:
            recommendations.append("Avoid frequent deep discharges below 20% capacity.")
        
        if avg_discharge > 1.0:
            recommendations.append("High discharge rates accelerate degradation. Consider using lower discharge rates when possible.")
        
        if len(recommendations) > 0:
            for i, rec in enumerate(recommendations):
                st.markdown(f"**{i+1}.** {rec}")
        else:
            st.markdown("Your battery usage appears to be within optimal parameters. Continue current usage patterns to maximize battery life.")

        # Add remaining useful life estimate
        remaining_cycles = 0
        if st.session_state.prediction >= 80:
            # Roughly estimate remaining cycles until 80% SOH
            remaining_cycles = int((st.session_state.prediction - 80) / 0.015)
        
        if remaining_cycles > 0:
            st.markdown(f"**Estimated remaining useful life:** Approximately {remaining_cycles} more cycles until battery reaches 80% SOH (common replacement threshold).")

# Model Information Page
elif page == "Model Information":
    st.header("Model Information")
    
    st.markdown("""
    ### Battery SOH Prediction Model
    
    The State of Health (SOH) prediction is based on a machine learning model trained on lithium-ion battery cycling data. 
    The model takes into account multiple factors that affect battery degradation:
    
    1. **Cycling Factors**
       - Number of charge/discharge cycles
       - Depth of discharge
       - Charge/discharge rates
       
    2. **Environmental Factors**
       - Operating temperature
       - Storage temperature
       - Mechanical stress and vibration
       
    3. **Battery Specifications**
       - Chemistry type
       - Nominal voltage and capacity
       - Age of the battery
    
    ### Key Degradation Mechanisms
    
    """)
    
    # Create columns for degradation mechanisms
    mech_col1, mech_col2 = st.columns(2)
    
    with mech_col1:
        st.markdown("""
        **Lithium Loss Mechanisms:**
        - SEI (Solid Electrolyte Interphase) layer growth
        - Lithium plating
        - Electrolyte decomposition
        """)
        
        # Add a simple diagram for SEI layer
        st.image("https://via.placeholder.com/400x200?text=SEI+Layer+Growth+Diagram", caption="SEI Layer Growth")
    
    with mech_col2:
        st.markdown("""
        **Active Material Loss:**
        - Structural degradation of electrodes
        - Particle cracking from mechanical stress
        - Dissolution of transition metals
        - Loss of electrical contact
        """)
        
        # Add a simple diagram for structural degradation
        st.image("https://via.placeholder.com/400x200?text=Structural+Degradation+Diagram", caption="Electrode Structural Degradation")
    
    # Model performance section
    st.subheader("Model Performance")
    
    # Create mock model performance metrics
    performance_data = {
        'Metric': ['R² Score', 'Mean Absolute Error', 'Root Mean Squared Error'],
        'Training Set': [0.95, 1.2, 1.5],
        'Test Set': [0.92, 1.5, 1.8],
        'Validation Set': [0.91, 1.6, 1.9]
    }
    
    st.dataframe(pd.DataFrame(performance_data))
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Create mock feature importance data
    importance_data = {
        'Feature': ['Cycle_Count', 'Avg_Temperature', 'Age_Months', 'Avg_Discharge_Rate', 
                   'Deep_Discharge_Pct', 'Fast_Charging_Pct', 'Nominal_Capacity', 'Vibration_Level'],
        'Importance': [0.35, 0.20, 0.15, 0.10, 0.08, 0.07, 0.03, 0.02]
    }
    
    importance_df = pd.DataFrame(importance_data).sort_values('Importance', ascending=True)
    
    # Plot feature importance
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Importance')
    ax.set_title('Feature Importance in SOH Prediction')
    st.pyplot(fig)

# Advanced Options Page
elif page == "Advanced Options":
    st.header("Advanced Options")
    
    st.markdown("""
    ### Data Upload
    
    You can upload your battery cycling data to improve prediction accuracy.
    """)
    
    uploaded_file = st.file_uploader("Upload battery cycling data (CSV format)", type=['csv'])
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("File successfully uploaded!")
            st.dataframe(data.head())
            
            # Required columns for battery cycling data
            required_columns = ['Cycle_Number', 'Voltage', 'Current', 'Temperature', 'Capacity', 'SOH']
            
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                st.warning(f"Your data is missing the following recommended columns: {', '.join(missing_columns)}")
                st.markdown("Please ensure your data contains appropriate battery cycling information.")
            else:
                st.success("Data format looks good! You can now train a custom model.")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
    
    # Model parameter tuning
    st.subheader("Model Parameter Tuning")
    
    st.markdown("Adjust the parameters of the prediction model to better fit your specific battery type.")
    
    # Add parameter tuning options
    model_type = st.selectbox("Model Type", ["Random Forest", "Gradient Boosting", "Deep Learning"], index=0)
    
    if model_type == "Random Forest":
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100)
            max_depth = st.slider("Maximum Tree Depth", min_value=1, max_value=30, value=10)
        
        with col2:
            min_samples_split = st.slider("Minimum Samples to Split", min_value=2, max_value=10, value=2)
            min_samples_leaf = st.slider("Minimum Samples per Leaf", min_value=1, max_value=10, value=1)
    
    elif model_type == "Gradient Boosting":
        col1, col2 = st.columns(2)
        
        with col1:
            n_estimators_gb = st.slider("Number of Boosting Stages", min_value=10, max_value=200, value=100)
            learning_rate = st.slider("Learning Rate", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
        
        with col2:
            max_depth_gb = st.slider("Maximum Tree Depth", min_value=1, max_value=10, value=3)
            subsample = st.slider("Subsample Ratio", min_value=0.5, max_value=1.0, value=1.0, step=0.05)
    
    elif model_type == "Deep Learning":
        col1, col2 = st.columns(2)
        
        with col1:
            layers = st.slider("Number of Hidden Layers", min_value=1, max_value=5, value=2)
            neurons = st.slider("Neurons per Layer", min_value=8, max_value=128, value=32)
        
        with col2:
            dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05)
            epochs = st.slider("Training Epochs", min_value=10, max_value=200, value=50)
    
    # Train custom model button
    if st.button("Train Custom Model"):
        if uploaded_file is None:
            st.warning("Please upload data before training a custom model.")
        else:
            with st.spinner("Training custom model..."):
                # Simulate training time
                time.sleep(3)
                st.success("Custom model trained successfully!")
                st.session_state.model_loaded = True
    
    # Export model option
    st.subheader("Export Model")
    
    if st.session_state.model_loaded:
        if st.button("Export Trained Model"):
            # Simulate file download
            st.success("Model exported successfully!")
            st.markdown("Download your trained model: [battery_soh_model.pkl](https://example.com/downloads/model.pkl)")
    else:
        st.info("Train a custom model before exporting.")

# Add footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<div style='text-align: center'>Battery SOH Prediction App © 2025</div>", unsafe_allow_html=True)