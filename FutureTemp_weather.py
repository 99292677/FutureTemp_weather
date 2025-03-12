import os
import pandas as pd
import numpy as np
import plotly.express as px
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st
import kagglehub  # ต้องติดตั้งด้วย pip install kagglehub

# ---------------------------
# ดาวน์โหลด dataset จาก Kaggle
# ---------------------------
# ดาวน์โหลด dataset โดยใช้ kagglehub
path = kagglehub.dataset_download("ananthr1/weather-prediction")
data_path = os.path.join(path, 'seattle-weather.csv')

# ---------------------------
# โหลดข้อมูลและเตรียมข้อมูล
# ---------------------------
data = pd.read_csv(data_path)
data.dropna(inplace=True)  # ลบค่า missing
data['date'] = pd.to_datetime(data['date'])  # แปลงคอลัมน์วันที่เป็น datetime

# Prepare Data for Training
training = data['temp_max'].values.reshape(-1, 1)

def df_to_XY(data_array, window_size=10):
    X, y = [], []
    for i in range(window_size, len(data_array)):
        X.append(data_array[i-window_size:i, 0])
        y.append(data_array[i, 0])
    return np.array(X), np.array(y)

WINDOW_SIZE = 10
X, y = df_to_XY(training, WINDOW_SIZE)

# แบ่งข้อมูลเป็น Train, Validation, Test
X_train, y_train = X[:800], y[:800]
X_val, y_val = X[800:1000], y[800:1000]
X_test, y_test = X[1000:], y[1000:]

# Reshape สำหรับ LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val   = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test  = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ----------------------------------------
# สร้าง/โหลดโมเดล LSTM (ถ้ายังไม่มีโมเดลที่บันทึกไว้)
# ----------------------------------------
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

# โหลดโมเดลที่บันทึกไว้
model = load_model('lstm_weather_model.h5')

# ---------------------------
# Streamlit App
# ---------------------------
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
window_size_input = st.sidebar.slider("Number of Days for Prediction", min_value=5, max_value=20, value=10)
inputs = []
for i in range(window_size_input):
    inputs.append(st.sidebar.number_input(f"Day {i+1} Temperature (°C):", value=10.0))

# Prediction and Metrics Section
if st.sidebar.button("🌡️ Predict Temperature"):
    input_data = np.array(inputs).reshape(1, -1, 1)
    prediction = model.predict(input_data)[0][0]

    # จำลอง actual temperature สำหรับคำนวณ metrics (ปรับปรุงใช้ข้อมูลจริงได้หากมี)
    actual_temp = [input_data[0, -1, 0] + np.random.uniform(-2, 2)]
    
    mae = mean_absolute_error(actual_temp, [prediction])
    rmse = np.sqrt(mean_squared_error(actual_temp, [prediction]))
    accuracy = 100 - (abs(actual_temp[0] - prediction) / abs(actual_temp[0]) * 100)

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
    df_plot.loc[len(df_plot)-1, 'Type'] = 'Prediction'
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
