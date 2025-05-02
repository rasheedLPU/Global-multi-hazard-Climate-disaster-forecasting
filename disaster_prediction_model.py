import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, concatenate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load the data
df = pd.read_csv('data/merged_natural_disaster_dataset_1992_2020 (1).csv')

# Define features and target variables
features = [
    'Country', 'Year', 'Temperature', 'Sea level Value',
    'Artificial surfaces (including urban and associated areas): Climate altering',
    'Grassland: Climate regulating', 'Woody crops: Climate regulating',
    'Terrestrial barren land: Climate neutral', 'Shrub-covered areas: Climate regulating',
    'Carbon stocks in forests', 'Forest area', 'Index of carbon stocks in forests',
    'Index of forest extent', 'Land area'
]

target = ['Total Disasters', 'Drought', 'Flood', 'Landslide', 'Storm', 'Wildfire']

# Preprocess the data
# Handle categorical variables
le = LabelEncoder()
df['Country_encoded'] = le.fit_transform(df['Country'])

# Save the label encoder
joblib.dump(le, 'label_encoder.joblib')

# Create feature matrix X
X = df[features].copy()
X['Country'] = df['Country_encoded']
y = df[target]

# Scale the features
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Save the scalers
joblib.dump(scaler_X, 'scaler_X.joblib')
joblib.dump(scaler_y, 'scaler_y.joblib')

# Reshape data for LSTM (samples, timesteps, features)
def create_sequences(X, y, time_steps=10):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# Build the attention-based LSTM model
def build_attention_model(input_shape, output_shape):
    inputs = Input(shape=input_shape)
    lstm1 = LSTM(64, return_sequences=True)(inputs)
    lstm2 = LSTM(32, return_sequences=True)(lstm1)
    attention = Attention()([lstm2, lstm2])
    flatten = tf.keras.layers.Flatten()(attention)
    dense1 = Dense(64, activation='relu')(flatten)
    dense2 = Dense(32, activation='relu')(dense1)
    outputs = Dense(output_shape, activation='linear')(dense2)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Create and compile the model
model = build_attention_model(input_shape=(time_steps, X_scaled.shape[1]), 
                            output_shape=len(target))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, 
                   epochs=50, 
                   batch_size=32,
                   validation_split=0.2,
                   verbose=1)

# Save the model
model.save('model.h5')

# Save the training history
joblib.dump(history.history, 'loss_history.joblib')

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform predictions and actual values
y_test_original = scaler_y.inverse_transform(y_test)
y_pred_original = scaler_y.inverse_transform(y_pred)

# Evaluation metrics
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))

# Save metrics
metrics = {
    'mae': mae,
    'rmse': rmse
}

joblib.dump(metrics, 'model_metrics.joblib')

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")

# Print individual metrics for each target variable
for i, target_name in enumerate(target):
    mae_i = mean_absolute_error(y_test_original[:, i], y_pred_original[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_test_original[:, i], y_pred_original[:, i]))
    print(f"\n{target_name}:")
    print(f"MAE: {mae_i:.2f}")
    print(f"RMSE: {rmse_i:.2f}")

# Visualization functions
def plot_loss_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curves.png')
    plt.close()

def plot_prediction_scatter(y_true, y_pred, target_names):
    n_targets = len(target_names)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (ax, target_name) in enumerate(zip(axes, target_names)):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5)
        ax.plot([y_true[:, i].min(), y_true[:, i].max()], 
                [y_true[:, i].min(), y_true[:, i].max()], 
                'r--', lw=2)
        ax.set_title(f'{target_name}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('prediction_scatter.png')
    plt.close()

# Generate visualizations
plot_loss_curves(history)
plot_prediction_scatter(y_test_original, y_pred_original, target)
