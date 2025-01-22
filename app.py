import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestRegressor

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('aqi_model.pkl')

# Add custom CSS for background image
def set_background_image(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set your background image URL or local path
background_image_url = "https://content.paulreiffer.com/wp-content/uploads/2015/10/Sunrise-Smog-Haze-Pollution-Kuala-Lumpur-KL-Malaysia-October-2015-Indonesia-Fires-Smoke-Sun-Paul-Reiffer-Photographer.jpg"  # Replace with your image URL
set_background_image(background_image_url)

# Initialize the Streamlit app
st.title("Air Quality Index Prediction")
st.markdown("""
This application allows users to explore air quality data, visualize pollutant trends, and predict AQI concentrations using a pre-trained Random Forest model.

### Malaysia AQI Categories:
- **0–50**: Good air quality
- **51–100**: Moderate air quality
- **101–200**: Unhealthy air quality
- **201–300**: Very unhealthy air quality
- **301–500**: Hazardous air quality
""")
# Sidebar menu
st.sidebar.header("Options")
menu = st.sidebar.radio("Select an option:", ("Upload Dataset", "Visualize Data", "Make Predictions"))

# Upload dataset section
if menu == "Upload Dataset":
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.write(data.head())
        
        # Save uploaded data for further use
        data.to_csv('uploaded_data.csv', index=False)
        st.success("Dataset uploaded successfully!")

# Visualize data section
if menu == "Visualize Data":
    st.header("Visualize Data")

    try:
        data = pd.read_csv('uploaded_data.csv')
        st.write("Dataset Preview:")
        st.write(data.head())

        # Convert 'date' column to datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])

        unique_pollutants = data['pollutant'].unique()

        # Plot pollutant concentration over time
        st.subheader("Pollutant Concentrations Over Time")
        plt.figure(figsize=(12, 6))
        for pollutant in unique_pollutants:
            subset = data[data['pollutant'] == pollutant]
            plt.plot(subset['date'], subset['concentration'], label=pollutant, alpha=0.7)

        plt.title('Pollutant Concentrations Over Time')
        plt.xlabel('Date')
        plt.ylabel('Concentration')
        plt.legend()
        plt.grid()
        st.pyplot(plt)

        # Plot distribution of pollutant concentrations
        st.subheader("Distribution of Pollutant Concentrations")
        plt.figure(figsize=(12, 8))
        for pollutant in unique_pollutants:
            subset = data[data['pollutant'] == pollutant]
            sns.kdeplot(subset['concentration'], label=pollutant, fill=True)

        plt.title('Distribution of Pollutant Concentrations')
        plt.xlabel('Concentration')
        plt.ylabel('Density')
        plt.legend()
        st.pyplot(plt)

    except FileNotFoundError:
        st.error("Please upload a dataset first in the 'Upload Dataset' section.")


# Make predictions section
# Make predictions section
if menu == "Make Predictions":
    st.header("Make Predictions")

    try:
        # Load model
        model = load_model()

        # User inputs for pollutants
        st.subheader("Enter Pollutant Concentrations for Prediction")
        co = st.number_input("CO (Carbon Monoxide) in µg/m³:", min_value=0.0, value=0.0, step=0.1)
        no2 = st.number_input("NO2 (Nitrogen Dioxide) in µg/m³:", min_value=0.0, value=0.0, step=0.1)
        o3 = st.number_input("O3 (Ozone) in µg/m³:", min_value=0.0, value=0.0, step=0.1)
        pm10 = st.number_input("PM10 (Particulate Matter 10) in µg/m³:", min_value=0.0, value=0.0, step=0.1)
        pm25 = st.number_input("PM2.5 (Particulate Matter 2.5) in µg/m³:", min_value=0.0, value=0.0, step=0.1)
        so2 = st.number_input("SO2 (Sulfur Dioxide) in µg/m³:", min_value=0.0, value=0.0, step=0.1)

        # Predict button
        if st.button("Make Prediction"):
            # Prepare the input data
            input_data = pd.DataFrame({
                'CO': [co],
                'NO2': [no2],
                'O3': [o3],
                'PM 10': [pm10],  # Include space as in the training data
                'PM 2.5': [pm25], # Include space as in the training data
                'SO2': [so2]
            })
            # Predict AQI
            aqi_prediction = model.predict(input_data)[0]

            # Categorize AQI based on Malaysia's AQI ranges
            def categorize_aqi_malaysia(aqi_value):
                if aqi_value <= 50:
                    return "Good", "Air quality is satisfactory and poses little or no risk."
                elif aqi_value <= 100:
                    return "Moderate", "Air quality is acceptable; some pollutants may be a concern for sensitive individuals."
                elif aqi_value <= 200:
                    return "Unhealthy", "Air quality may cause adverse health effects for sensitive groups."
                elif aqi_value <= 300:
                    return "Very Unhealthy", "Health alert: everyone may experience serious health effects."
                else:
                    return "Hazardous", "Emergency conditions: health warnings for all population groups."

            # Categorize the AQI prediction
            category, description = categorize_aqi_malaysia(aqi_prediction)

            # Display the results
            st.subheader(f"Predicted AQI: {aqi_prediction:.1f}")
            st.markdown(f"### {category}")
            st.write(description)

    except Exception as e:
        st.error(f"An error occurred: {e}")
