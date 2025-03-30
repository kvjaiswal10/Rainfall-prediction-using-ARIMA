import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima

df = pd.read_csv(r"C:\Users\kvjai\ML PROJECTS\Rainfall prediction - ARIMA\data\Rainfall_data.csv") 

df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df.sort_values(by='Date')

train_size = int(len(df) * 0.8)
train = df['Precipitation'][:train_size]
test = df['Precipitation'][train_size:]

model = auto_arima(train,
                   start_p=0, max_p=5,
                   start_q=0, max_q=5,
                   d=None, max_d=2,
                   seasonal=False,  # True if seasonality
                   trace=True,
                   error_action='ignore',
                   suppress_warnings=True,
                   stepwise=True)

print("\nBest ARIMA Order Found:", model.order)

model.fit(train)

# forecast
n_periods = len(test)
forecast = model.predict(n_periods=n_periods)

# actual vs prediction
plt.figure(figsize=(12, 6))
plt.plot(range(len(train)), train, label='Train', color='blue')
plt.plot(range(len(train), len(train) + len(test)), test, label='Actual (Test)', color='green')
plt.plot(range(len(train), len(train) + len(test)), forecast, label='Forecast', color='red', linestyle='--')
plt.title("Actual vs Forecasted Rainfall (Auto ARIMA)")
plt.xlabel("Time Index")
plt.ylabel("Precipitation (mm)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# evaluation
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {round(rmse, 2)}")

