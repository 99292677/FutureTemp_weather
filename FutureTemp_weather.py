import os
import pandas as pd
import numpy as np
import plotly.express as px
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st

# Set Kaggle API
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# Download the dataset using Kaggle API if not already downloaded
if not os.path.exists("seattle-weather.csv"):
    os.system("kaggle datasets download -d ananthr1/weather-prediction --unzip")

# Load the dataset
data = pd.read_csv('seattle-weather.csv')
data.dropna(inplace=True)  # Remove missing values
data['date'] = pd.to_datetime(data['date'])  # Convert date to datetime

# Prepare Data for Training
training = data['temp_max'].values.reshape(-1, 1)

def df_to_XY(data, window_size=10):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

WINDOW_SIZE = 10
X, y = df_to_XY(training, WINDOW_SIZE)

# Split into Train, Validation, Test
X_train, y_train = X[:800], y[:800]
X_val, y_val = X[800:1000], y[800:1000]
X_test, y_test = X[1000:], y[1000:]

# Reshape for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define and Train LSTM Model if not already saved
if not os.path.exists('lstm_weather_model.h5'):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
    model.save('lstm_weather_model.h5')

# Load the trained model
model = load_model('lstm_weather_model.h5')

# Streamlit App Code
st.set_page_config(page_title="Seattle Weather Predictor", page_icon="🌤️", layout="wide")

# Title Section
st.title("🌤️ FutureTemp Weather Predictor")
st.markdown("""
Welcome to the **FutureTemp**! This tool uses **LSTM Neural Networks** to forecast the temperature based on historical data. 
Enjoy a visually appealing and interactive experience. 🚀
""")

# Sidebar Section for Input
st.sidebar.header("🔧 Configure Inputs")
st.sidebar.markdown("Adjust the input parameters below:")
window_size = st.sidebar.slider("Number of Days for Prediction", min_value=5, max_value=20, value=10)
inputs = []
for i in range(window_size):
    inputs.append(st.sidebar.number_input(f"Day {i+1} Temperature (°C):", value=10.0))

# Prediction and Metrics Section
if st.sidebar.button("🌡️ Predict Temperature"):
    input_data = np.array(inputs).reshape(1, -1, 1)
    prediction = model.predict(input_data)[0][0]

    # Simulate actual temperature for metrics (replace with real data if available)
    actual_temp = [input_data[0, -1, 0] + np.random.uniform(-2, 2)]

    # Calculate Metrics
    mae = mean_absolute_error(actual_temp, [prediction])
    rmse = np.sqrt(mean_squared_error(actual_temp, [prediction]))
    accuracy = 100 - (abs(actual_temp[0] - prediction) / abs(actual_temp[0]) * 100)

    # Result Cards
    st.markdown("## 📊 Results")
    st.success(f"🌡️ **Predicted Temperature**: {prediction:.2f} °C")
    st.info(f"📏 **Simulated Actual Temperature**: {actual_temp[0]:.2f} °C")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f} °C")
    with col2:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} °C")
    with col3:
        st.metric("Prediction Accuracy (%)", f"{accuracy:.2f} %")

    # Graph Section
    st.markdown("## 📈 Temperature Visualization")
    df_plot = pd.DataFrame({
        'Day': range(len(inputs) + 1),
        'Temperature': inputs + [actual_temp[0]],
        'Type': ['Input'] * len(inputs) + ['Actual']
    })
    df_plot.loc[len(df_plot) - 1, 'Type'] = 'Prediction'

    fig = px.line(df_plot, x='Day', y='Temperature', color='Type',
                  title="Temperature Predictions vs Actual",
                  labels={'Temperature': 'Temperature (°C)', 'Day': 'Day'},
                  template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Footer Section
st.markdown("""
    ---
    Made with ❤️ by **by Boss 👦🏻 Ice 🧊 Film 🎞️**  
    """)
