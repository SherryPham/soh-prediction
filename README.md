# Battery State of Health (SOH) Prediction App

## Overview

This Streamlit application predicts the State of Health (SOH) of lithium-ion batteries based on operating parameters and battery specifications. Battery SOH is a crucial metric that indicates the condition of a battery relative to its ideal conditions, helping users determine when maintenance or replacement might be necessary.

## Features

- **Simple Input Interface**: Enter battery parameters such as cycle count, operating temperature, voltage, and other specifications
- **Instant SOH Prediction**: Get immediate predictions of battery health based on input parameters
- **Visual Results**: View predictions with an intuitive gauge chart visualization
- **Actionable Recommendations**: Receive recommendations to improve battery life based on input parameters
- **Technical Information**: Access detailed information about battery degradation mechanisms and model details

## How to Use

1. **Input Battery Parameters**: Enter your battery's specifications including:
   - Cycle count
   - Battery age
   - Operating temperature
   - Nominal voltage and capacity
   - Battery chemistry
   - Discharge rates
   - Additional operational factors

2. **Generate Prediction**: Click the "Predict Battery SOH" button

3. **Review Results**: The app will display:
   - Numerical SOH prediction (percentage)
   - Visual gauge representation
   - Battery status (Excellent/Good/Fair/Poor)
   - Specific recommendations for optimizing battery life

## Installation and Local Setup

### Prerequisites
- Python 3.8 or higher

### Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/SherryPham/COS40005.git
   cd COS40005
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # Activate on Windows
   venv\Scripts\activate
   
   # Activate on macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to `http://localhost:8501`

## Deployment

This app is deployed on Streamlit Cloud and can be accessed at: [https://soh-prediction-5olaajfhgkvjd8kyk8in6s.streamlit.app/](https://soh-prediction-app/)

## Model Information

The prediction model takes into account various factors affecting battery health:

- **Cycling Factors**: Number of cycles, depth of discharge, charge/discharge rates
- **Environmental Factors**: Operating temperature, storage conditions, mechanical stress
- **Battery Specifications**: Chemistry type, nominal voltage/capacity, age

## Technical Background

Lithium-ion battery degradation typically occurs through:

1. **Lithium Loss Mechanisms**:
   - SEI (Solid Electrolyte Interphase) layer growth
   - Lithium plating
   - Electrolyte decomposition

2. **Active Material Loss**:
   - Structural degradation of electrodes
   - Particle cracking from mechanical stress
   - Dissolution of transition metals
   - Loss of electrical contact


## License

This project is licensed under the MIT License 


