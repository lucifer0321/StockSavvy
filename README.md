# StockSavvy
StockSavvy is an AI-driven stock market app with real-time updates, forecasting (LSTM, ARIMA, XGBoost), sentiment analysis (FinBERT, VADER), and a pseudo-trading simulator.

code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load your dataset
df = pd.read_csv('NIFTY50_10yearsdata1.csv')  # replace with your file
df.columns = df.columns.str.strip()
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# 2. Use only 'Close' prices
data = df['Close'].values.reshape(-1, 1)

# 3. Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# 4. Create sequences for LSTM
def create_dataset(dataset, time_step=60, predict_days=7):
    x, y = [], []
    for i in range(time_step, len(dataset) - predict_days + 1):
        x.append(dataset[i-time_step:i, 0])
        y.append(dataset[i:i+predict_days, 0])  # next 7 days
    return np.array(x), np.array(y)

time_step = 60
predict_days = 7
x, y = create_dataset(scaled_data, time_step, predict_days)
x = x.reshape((x.shape[0], x.shape[1], 1))  # (samples, timesteps, features)

# 5. Split into Train and Test
train_size = int(len(x) * 0.8)  # 80% train, 20% test
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 6. Build the LSTM Model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    LSTM(64),
    Dense(predict_days)  # output 7 future days
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 7. Train the model
early_stop = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
model.fit(x_train, y_train, epochs=50, batch_size=64, callbacks=[early_stop], verbose=1)

# 8. Predict next 7 days
last_60_days = scaled_data[-time_step:]
last_60_days = last_60_days.reshape((1, time_step, 1))
predicted_next_7 = model.predict(last_60_days)
predicted_next_7 = scaler.inverse_transform(predicted_next_7).flatten()

# 9. Show predicted prices
print("\n🔮 Predicted next 7 closing prices:")
for i, price in enumerate(predicted_next_7, 1):
    print(f"Day {i}: ₹{price:.2f}")

# 10. Plot actual vs predicted on test set
y_test_pred = model.predict(x_test)
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test_actual = scaler.inverse_transform(y_test)

# For simplicity, plot the first day's prediction only
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[:, 0], label='Actual Price (Day 1 Prediction)')
plt.plot(y_test_pred[:, 0], label='Predicted Price (Day 1 Prediction)')
plt.title('LSTM - Actual vs Predicted (First Day Prediction)')
plt.xlabel('Days')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()




