import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import pickle
from pathlib import Path
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create directories if they don't exist
current_dir = Path(__file__).parent.absolute()
models_dir = current_dir.parent / 'models'
data_dir = current_dir.parent / 'data'
models_dir.mkdir(parents=True, exist_ok=True)

# Custom Attention Layer
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

# Load and preprocess data
print("Loading data...")
df = pd.read_csv(data_dir / 'merged_natural_disaster_dataset_1992_2020 (1).csv')

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

# Prepare data
print("Preparing data...")
X = df[features].copy()
y = df[target].copy()

# Encode categorical variables
le = LabelEncoder()
X['Country'] = le.fit_transform(X['Country'])

# Scale features and targets
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Create sequences for LSTM
sequence_length = 5
X_sequences = []
y_sequences = []

for i in range(len(X_scaled) - sequence_length):
    X_sequences.append(X_scaled[i:i + sequence_length])
    y_sequences.append(y_scaled[i + sequence_length])

X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)

# Split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sequences, y_sequences, test_size=0.2, random_state=42
)

# Build model
print("Building and training LSTM model...")
input_shape = (sequence_length, X.shape[1])

inputs = Input(shape=input_shape)
lstm1 = LSTM(128, return_sequences=True)(inputs)
lstm2 = LSTM(64, return_sequences=True)(lstm1)
attention = AttentionLayer()(lstm2)
dense1 = Dense(64, activation='relu')(attention)
outputs = Dense(len(target), activation='sigmoid')(dense1)

model = Model(inputs=inputs, outputs=outputs)

# Compile model with a lower learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    str(models_dir / 'best_model.h5'),
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001,
    verbose=1
)

# Train model with improved parameters
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=200,  # Increased from 100 to 200 epochs
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
).history

print("\nTraining history keys:", list(history.keys()))

# Load the best model
model = tf.keras.models.load_model(
    models_dir / 'best_model.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)

# Make predictions
print("Making predictions...")
y_pred = model.predict(X_test)

# Calculate metrics
print("Calculating metrics...")
metrics = {}
validation_data = {
    'X_test': X_test,
    'y_test': y_test,
    'y_pred': y_pred,
    'history': history
}

print("\nModel Performance Metrics:")
for i, target_name in enumerate(target):
    mae_i = mean_absolute_error(y_test[:, i], y_pred[:, i])
    rmse_i = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
    # Calculate R-squared score
    r2_i = 1 - np.sum((y_test[:, i] - y_pred[:, i])**2) / np.sum((y_test[:, i] - np.mean(y_test[:, i]))**2)
    
    metrics[target_name] = {
        'mae': float(mae_i),
        'rmse': float(rmse_i),
        'r2': float(r2_i)
    }
    print(f"\n{target_name}:")
    print(f"  MAE: {mae_i:.4f}")
    print(f"  RMSE: {rmse_i:.4f}")
    print(f"  R²: {r2_i:.4f}")

# Calculate average metrics across all targets
avg_mae = np.mean([m['mae'] for m in metrics.values()])
avg_rmse = np.mean([m['rmse'] for m in metrics.values()])
avg_r2 = np.mean([m['r2'] for m in metrics.values()])

metrics['overall'] = {
    'avg_mae': float(avg_mae),
    'avg_rmse': float(avg_rmse),
    'avg_r2': float(avg_r2)
}

print("\nOverall Model Performance:")
print(f"Average MAE: {avg_mae:.4f}")
print(f"Average RMSE: {avg_rmse:.4f}")
print(f"Average R²: {avg_r2:.4f}")

# Save all necessary files
print("\nSaving model and metrics...")

# Save validation data using pickle
try:
    with open(models_dir / 'validation_data.pkl', 'wb') as f:
        pickle.dump(validation_data, f)
    print("Successfully saved validation_data.pkl")
except Exception as e:
    print("Error saving validation_data:", str(e))

# Save other files
model.save(models_dir / 'disaster_prediction_model.h5')
joblib.dump(scaler_X, models_dir / 'scaler_X.joblib')
joblib.dump(scaler_y, models_dir / 'scaler_y.joblib')
joblib.dump(le, models_dir / 'label_encoder.joblib')
joblib.dump(metrics, models_dir / 'model_metrics.joblib')

print("\nModel training completed. All files saved in the models directory.")
