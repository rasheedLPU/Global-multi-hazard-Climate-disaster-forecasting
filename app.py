import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
import joblib
import os

# Set page configuration
st.set_page_config(page_title="Disaster Prediction Dashboard", layout="wide")

# Title
st.title("Natural Disaster Prediction Dashboard")

# Sidebar
st.sidebar.header("Model Information")
st.sidebar.write("This dashboard shows predictions and analysis of natural disasters using an attention-based LSTM model.")

# Load the data and model
@st.cache_data
def load_data():
    df = pd.read_csv('data/merged_natural_disaster_dataset_1992_2020 (1).csv')
    return df

# Load the model and scalers
@st.cache_resource
def load_model_and_scalers():
    if os.path.exists('model.h5'):
        model = load_model('model.h5')
        scaler_X = joblib.load('scaler_X.joblib')
        scaler_y = joblib.load('scaler_y.joblib')
        le = joblib.load('label_encoder.joblib')
        return model, scaler_X, scaler_y, le
    else:
        st.error("Model files not found. Please train the model first.")
        return None, None, None, None

# Load data and model
df = load_data()
model, scaler_X, scaler_y, le = load_model_and_scalers()

# Features and targets
features = [
    'Country', 'Year', 'Temperature', 'Sea level Value',
    'Artificial surfaces (including urban and associated areas): Climate altering',
    'Grassland: Climate regulating', 'Woody crops: Climate regulating',
    'Terrestrial barren land: Climate neutral', 'Shrub-covered areas: Climate regulating',
    'Carbon stocks in forests', 'Forest area', 'Index of carbon stocks in forests',
    'Index of forest extent', 'Land area'
]

target = ['Total Disasters', 'Drought', 'Flood', 'Landslide', 'Storm', 'Wildfire']

# Main content
st.header("Disaster Prediction Analysis")

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["Predictions", "Model Performance", "Data Analysis"])

with tab1:
    st.subheader("Make Predictions")
    
    # Country selection
    country = st.selectbox("Select Country", df['Country'].unique())
    
    # Year selection
    year = st.slider("Select Year", min_value=2020, max_value=2025, value=2023)
    
    if st.button("Predict"):
        # Get the last 5 years of data for the selected country
        country_data = df[df['Country'] == country].sort_values('Year', ascending=False).head(5)
        
        if len(country_data) >= 5:
            # Prepare input data
            X = country_data[features].copy()
            X['Country'] = le.transform(X['Country'])
            X_scaled = scaler_X.transform(X)
            X_seq = np.expand_dims(X_scaled, axis=0)
            
            # Make prediction
            prediction = model.predict(X_seq)
            prediction_original = scaler_y.inverse_transform(prediction)
            
            # Display predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Predicted Natural Disasters for", year)
                pred_df = pd.DataFrame(prediction_original[0], index=target, columns=['Predicted Count'])
                st.dataframe(pred_df)
            
            with col2:
                # Create bar chart
                fig = px.bar(pred_df, orientation='h')
                fig.update_layout(title=f"Predicted Disasters for {country} in {year}")
                st.plotly_chart(fig)
        else:
            st.error("Insufficient historical data for prediction")

with tab2:
    st.subheader("Model Performance")
    
    # Load and display metrics
    if os.path.exists('model_metrics.joblib'):
        metrics = joblib.load('model_metrics.joblib')
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Overall MAE", f"{metrics['mae']:.2f}")
            st.metric("Overall RMSE", f"{metrics['rmse']:.2f}")
        
        # Display loss curves
        if os.path.exists('loss_history.joblib'):
            history = joblib.load('loss_history.joblib')
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history['loss'], name='Training Loss'))
            fig.add_trace(go.Scatter(y=history['val_loss'], name='Validation Loss'))
            fig.update_layout(title='Model Loss Curves', xaxis_title='Epoch', yaxis_title='Loss')
            st.plotly_chart(fig)

with tab3:
    st.subheader("Historical Data Analysis")
    
    # Time series analysis
    st.write("Historical Disaster Trends")
    
    # Select disaster type
    disaster_type = st.selectbox("Select Disaster Type", target)
    
    # Group by year and calculate mean
    yearly_data = df.groupby('Year')[disaster_type].mean().reset_index()
    
    # Create line chart
    fig = px.line(yearly_data, x='Year', y=disaster_type, 
                  title=f'Average {disaster_type} Over Time')
    st.plotly_chart(fig)
    
    # Correlation analysis
    st.write("Feature Correlation Analysis")
    correlation = df[features + target].corr()
    fig = px.imshow(correlation, title='Feature Correlation Matrix')
    st.plotly_chart(fig)

# Footer
st.markdown("---")
st.markdown("Dashboard created with Streamlit for Natural Disaster Prediction")
