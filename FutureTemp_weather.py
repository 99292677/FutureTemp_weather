import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ตรวจสอบเวอร์ชัน Python
if sys.version_info.major == 3 and sys.version_info.minor >= 13:
    sys.exit("TensorFlow ยังไม่รองรับ Python 3.13 กรุณาใช้ Python 3.9 หรือ 3.10 แทน")

try:
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, LSTM, Dropout
except ModuleNotFoundError:
    sys.exit("ไม่พบโมดูล 'tensorflow' กรุณาติดตั้ง TensorFlow ด้วยคำสั่ง: pip install tensorflow (โดยใช้ Python เวอร์ชันที่รองรับ)")

# ตั้งค่า Kaggle API
if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
    os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
    os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]

# ใช้ caching สำหรับการโหลดข้อมูล
@st.cache_data(show_spinner=False)
def load_data():
    if not os.path.exists("seattle-weather.csv"):
        os.system("kaggle datasets download -d ananthr1/weather-prediction --unzip")
    data = pd.read_csv('seattle-weather.csv')
    data.dropna(inplace=True)
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()

# เตรียมข้อมูลสำหรับการเทรน
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

# ปรับขนาดข้อมูลสำหรับ LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# ใช้ caching สำหรับการโหลดหรือเทรนโมเดล
@st.cache_resource(show_spinner=False)
def get_model():
    if os.path.exists('lstm_weather_model.h5'):
        model = load_model('lstm_weather_model.h5')
    else:
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
        model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)
        model.save('lstm_weather_model.h5')
    return model

model = get_model()

# ตั้งค่า Streamlit Page
st.set_page_config(page_title="Seattle Weather Predictor", page_icon="🌤️", layout="wide")

# ส่วน Title
st.title("🌤️ FutureTemp Weather Predictor")
st.markdown("""
Welcome to the **FutureTemp**! This tool uses **LSTM Neural Networks** to forecast the temperature based on historical data. 
Enjoy a visually appealing and interactive experience. 🚀
""")

# Sidebar สำหรับ Input
st.sidebar.header("🔧 Configure Inputs")
st.sidebar.markdown("Adjust the input parameters below:")
window_size_input = st.sidebar.slider("Number of Days for Prediction", min_value=5, max_value=20, value=10)
inputs = []
for i in range(window_size_input):
    inputs.append(st.sidebar.number_input(f"Day {i+1} Temperature (°C):", value=10.0))

# ส่วนการคำนวณและแสดงผล
if st.sidebar.button("🌡️ Predict Temperature"):
    input_data = np.array(inputs).reshape(1, -1, 1)
    prediction = model.predict(input_data)[0][0]

    # จำลองค่า actual temperature สำหรับการคำนวณ metrics (เปลี่ยนได้หากมีข้อมูลจริง)
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

st.markdown("""
    ---
    Made with ❤️ by **by Boss 👦🏻 Ice 🧊 Film 🎞️**
    """)
