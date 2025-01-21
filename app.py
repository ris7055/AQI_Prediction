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

# Initialize the Streamlit app
st.title("Air Quality Index Prediction")
st.markdown("This application allows users to explore air quality data, visualize pollutant trends, and predict AQI concentrations using a pre-trained Random Forest model.")

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
if menu == "Make Predictions":
    st.header("Make Predictions")

    try:
        # Load model and data
        model = load_model()
        data = pd.read_csv('uploaded_data.csv')

        # Ensure required features are present
        required_features = ['year', 'month', 'day', 'lag_1', 'lag_2', 'rolling_mean_3']
        for feature in required_features:
            if feature not in data.columns:
                st.error(f"The dataset must contain the following column: {feature}")
                st.stop()

        # User inputs for prediction
        st.subheader("Enter Data for Prediction")
        year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        day = st.number_input("Day", min_value=1, max_value=31, value=1)
        lag_1 = st.number_input("Lag 1 (Previous Day's Concentration)", value=0.0)
        lag_2 = st.number_input("Lag 2 (Two Days Ago's Concentration)", value=0.0)
        rolling_mean_3 = st.number_input("Rolling Mean (3-Day Average)", value=0.0)

        # Prepare the input data
        input_data = pd.DataFrame({
            'year': [year],
            'month': [month],
            'day': [day],
            'lag_1': [lag_1],
            'lag_2': [lag_2],
            'rolling_mean_3': [rolling_mean_3]
        })

        # Predict concentration
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Concentration: {prediction:.2f}")

    except FileNotFoundError:
        st.error("Please upload a dataset first in the 'Upload Dataset' section.")

    except Exception as e:
        st.error(f"An error occurred: {e}")