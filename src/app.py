import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
import joblib
import pickle
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from pathlib import Path
import os

# Define the AttentionLayer class exactly as in train.py
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                               shape=(input_shape[-1], 1),
                               initializer='random_normal',
                               trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W))
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Get the current directory and set up paths
current_dir = Path(__file__).parent.absolute()
models_dir = current_dir.parent / 'models'
data_dir = current_dir.parent / 'data'

# Set page configuration
st.set_page_config(
    page_title="Natural Disaster Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the dashboard appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 5px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# Title with custom styling
st.title("🌍 Natural Disaster Prediction Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This dashboard uses an Attention-based LSTM model to predict natural disasters.
    The model considers various environmental and geographical factors to make predictions.
    """)
    
    st.header("Model Information")
    st.write("Features used:")
    st.write("- Environmental factors")
    st.write("- Geographical data")
    st.write("- Historical disaster records")

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv(data_dir / 'merged_natural_disaster_dataset_1992_2020 (1).csv')

@st.cache_resource
def load_model_and_scalers():
    # Load model with custom objects
    model = tf.keras.models.load_model(
        models_dir / 'disaster_prediction_model.h5',
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    scaler_X = joblib.load(models_dir / 'scaler_X.joblib')
    scaler_y = joblib.load(models_dir / 'scaler_y.joblib')
    le = joblib.load(models_dir / 'label_encoder.joblib')
    metrics = joblib.load(models_dir / 'model_metrics.joblib')
    
    try:
        with open(models_dir / 'validation_data.pkl', 'rb') as f:
            validation_data = pickle.load(f)
        print("\nLoaded validation_data keys:", list(validation_data.keys()))
        if 'history' in validation_data:
            print("History type:", type(validation_data['history']))
            print("History keys:", list(validation_data['history'].keys()))
    except Exception as e:
        print("\nError loading validation_data:", str(e))
        validation_data = {
            'X_test': None,
            'y_test': None,
            'y_pred': None
        }
    
    return model, scaler_X, scaler_y, le, metrics, validation_data

# Load the data and model
try:
    df = load_data()
    model, scaler_X, scaler_y, le, metrics, validation_data = load_model_and_scalers()
    
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
    tab1, tab2, tab3 = st.tabs(["🎯 Predictions", "📊 Model Performance", "📈 Data Analysis"])

    with tab1:
        st.header("Disaster Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            country = st.selectbox("Select Country", df['Country'].unique())
            year = st.slider("Select Year for Prediction", 2020, 2030, 2023)
            
        if st.button("Generate Predictions", key="predict_button"):
            with st.spinner("Generating predictions..."):
                # Get latest data for the country
                country_data = df[df['Country'] == country].sort_values('Year', ascending=False).iloc[0].copy()
                
                # Update the year in the data
                country_data['Year'] = year
                
                # Prepare input data
                X = pd.DataFrame([country_data[features].copy()])
                X['Country'] = le.transform([country])
                X_scaled = scaler_X.transform(X)
                
                # Create sequence for LSTM with temporal progression
                base_years = np.arange(year-4, year+1)  # Create 5 consecutive years leading to prediction year
                X_seq = []
                
                for yr in base_years:
                    temp_data = X.copy()
                    temp_data['Year'] = yr
                    X_seq.append(scaler_X.transform(temp_data))
                
                X_seq = np.array(X_seq)  # Shape: (5, 1, 14)
                X_seq = X_seq.reshape(1, 5, 14)  # Reshape to (1, 5, 14) for model input
                
                # Make prediction
                prediction = model.predict(X_seq)
                prediction_original = scaler_y.inverse_transform(prediction)
                
                # Display predictions
                st.subheader(f"Predicted Natural Disasters for {country} in {year}")
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'Disaster Type': target,
                    'Predicted Count': prediction_original[0]
                }).reset_index(drop=True)
                
                # Display as a nice table
                st.dataframe(
                    pred_df.style.background_gradient(subset=['Predicted Count'], cmap='YlOrRd')
                )
                
                # Create interactive bar chart
                fig = px.bar(pred_df, 
                            x='Disaster Type', 
                            y='Predicted Count',
                            title=f'Predicted Disasters for {country} in {year}',
                            color='Predicted Count',
                            color_continuous_scale='Viridis')
                
                fig.update_layout(
                    xaxis_title="Disaster Type",
                    yaxis_title="Predicted Number of Events",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.header("Model Performance Analysis")
        
        # Display metrics
        st.subheader("Model Accuracy Metrics")
        
        # Get validation data and training history
        X_test = validation_data['X_test']
        y_test = validation_data['y_test']
        y_pred = validation_data['y_pred']
        
        # Check if training history is available
        if 'history' in validation_data:
            history = validation_data['history']
            
            # Plot training history
            st.subheader("Training History")
            col1, col2 = st.columns(2)
            
            with col1:
                # Plot loss
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['loss'],
                    name='Training Loss',
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_loss'],
                    name='Validation Loss',
                    mode='lines',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title='Model Loss During Training',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Plot MAE
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=history['mae'],
                    name='Training MAE',
                    mode='lines',
                    line=dict(color='blue')
                ))
                fig.add_trace(go.Scatter(
                    y=history['val_mae'],
                    name='Validation MAE',
                    mode='lines',
                    line=dict(color='red')
                ))
                fig.update_layout(
                    title='Model MAE During Training',
                    xaxis_title='Epoch',
                    yaxis_title='MAE',
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Training history not available. Run train.py to see training progress visualization.")
        
        # Create metrics table
        metrics_df = pd.DataFrame(columns=['Disaster Type', 'MAE', 'RMSE', 'Recent MAE', 'Recent RMSE'])
        
        # Calculate recent metrics using last 50 samples
        recent_start = max(0, len(y_test) - 50)
        y_test_recent = y_test[recent_start:]
        y_pred_recent = y_pred[recent_start:]
        
        for i, disaster_type in enumerate(target):
            mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
            rmse_i = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            
            # Calculate recent metrics
            mae_recent = mean_absolute_error(y_test_recent[:, i], y_pred_recent[:, i])
            rmse_recent = np.sqrt(mean_squared_error(y_test_recent[:, i], y_pred_recent[:, i]))
            
            new_row = pd.DataFrame({
                'Disaster Type': [disaster_type],
                'MAE': [mae_i],
                'RMSE': [rmse_i],
                'Recent MAE': [mae_recent],
                'Recent RMSE': [rmse_recent]
            })
            metrics_df = pd.concat([metrics_df, new_row], ignore_index=True)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Overall Performance")
            metrics_df_overall = metrics_df[['Disaster Type', 'MAE', 'RMSE']].copy()
            st.dataframe(
                metrics_df_overall.style.background_gradient(subset=['MAE', 'RMSE'], cmap='RdYlGn_r')
            )
        
        with col2:
            st.subheader("Recent Performance (Last 50 Predictions)")
            metrics_df_recent = metrics_df[['Disaster Type', 'Recent MAE', 'Recent RMSE']].copy()
            st.dataframe(
                metrics_df_recent.style.background_gradient(subset=['Recent MAE', 'Recent RMSE'], cmap='RdYlGn_r')
            )
        
        # Create bar chart comparing overall and recent metrics
        st.subheader("Performance Comparison")
        fig = go.Figure()
        
        # Add overall metrics
        fig.add_trace(go.Bar(
            name='Overall MAE',
            x=metrics_df['Disaster Type'],
            y=metrics_df['MAE'],
            marker_color='blue',
            opacity=0.6
        ))
        
        fig.add_trace(go.Bar(
            name='Recent MAE',
            x=metrics_df['Disaster Type'],
            y=metrics_df['Recent MAE'],
            marker_color='red',
            opacity=0.6
        ))
        
        fig.update_layout(
            title='Model Performance: Overall vs Recent',
            barmode='group',
            xaxis_title="Disaster Type",
            yaxis_title="Mean Absolute Error (MAE)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Add performance insights
        st.subheader("Performance Insights")
        for disaster_type in target:
            row = metrics_df[metrics_df['Disaster Type'] == disaster_type].iloc[0]
            mae_diff = row['Recent MAE'] - row['MAE']
            if abs(mae_diff) > 0.1:  # Only show significant changes
                trend = "improved" if mae_diff < 0 else "declined"
                st.write(f"- {disaster_type} prediction has {trend} recently: MAE change of {mae_diff:.3f}")

    with tab3:
        st.header("Historical Data Analysis")
        
        # Time series analysis
        st.subheader("Disaster Trends Over Time")
        
        disaster_type = st.selectbox("Select Disaster Type", target)
        selected_country = st.selectbox("Select Country for Trend Analysis", df['Country'].unique(), key='trend_country')
        
        # Filter data for selected country and create yearly aggregates
        country_data = df[df['Country'] == selected_country]
        yearly_data = country_data.groupby('Year')[disaster_type].sum().reset_index()
        
        fig = px.line(yearly_data, 
                     x='Year', 
                     y=disaster_type,
                     title=f'{disaster_type} Occurrences in {selected_country} Over Time',
                     markers=True)
        fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Number of Events",
            showlegend=False,
            hovermode='x unified'
        )
        fig.update_traces(line_width=2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add summary statistics
        st.subheader("Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Events per Year", 
                     f"{yearly_data[disaster_type].mean():.2f}")
        with col2:
            st.metric("Maximum Events", 
                     f"{yearly_data[disaster_type].max():.0f}",
                     f"Year: {yearly_data.loc[yearly_data[disaster_type].idxmax(), 'Year']}")
        with col3:
            recent_trend = yearly_data[disaster_type].iloc[-1] - yearly_data[disaster_type].iloc[-2]
            st.metric("Recent Trend", 
                     f"{yearly_data[disaster_type].iloc[-1]:.0f}",
                     f"{recent_trend:+.1f} from previous year")
        
        # Correlation analysis
        st.subheader("Feature Correlation Analysis")
        
        # Select relevant features for correlation
        correlation_features = [
            'Temperature', 'Sea level Value',
            'Carbon stocks in forests', 'Forest area',
            'Index of forest extent', 'Land area'
        ]
        
        # Calculate correlation for the selected country
        correlation_data = country_data[correlation_features + [disaster_type]]
        correlation = correlation_data.corr()
        
        fig = px.imshow(correlation,
                       title=f'Feature Correlation Matrix for {selected_country}',
                       color_continuous_scale='RdBu',
                       aspect='auto')
        fig.update_layout(
            height=500,
            width=700
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Geographical distribution
        st.subheader("Geographical Distribution of Disasters")
        year_select = st.slider("Select Year for Map", int(df['Year'].min()), int(df['Year'].max()))
        
        yearly_country_data = df[df['Year'] == year_select].groupby('Country')[target].sum().reset_index()
        
        disaster_select = st.selectbox("Select Disaster Type for Map", target, key='map_disaster')
        fig = px.choropleth(yearly_country_data,
                           locations='Country',
                           locationmode='country names',
                           color=disaster_select,
                           title=f'{disaster_select} Distribution by Country ({year_select})',
                           color_continuous_scale='Viridis')
        fig.update_layout(
            height=600,
            geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
            margin={"r":0,"t":30,"l":0,"b":0}
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"""
    Error loading the model or data. Please ensure:
    1. The model has been trained (run train.py first)
    2. All required files are in the correct locations
    3. The dataset is available in the data directory
    
    Error details: {str(e)}
    """)
