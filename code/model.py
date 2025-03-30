import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv(r"C:\Users\kvjai\ML PROJECTS\Rainfall prediction - ARIMA\data\Rainfall_data.csv") 

df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df.sort_values(by='Date')

# aggreagate to month wise data
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_rainfall = df.groupby('YearMonth')['Precipitation'].sum().reset_index()
monthly_rainfall['YearMonth'] = monthly_rainfall['YearMonth'].astype(str)

rainfall_series = monthly_rainfall['Precipitation']

# stationarity check
result = adfuller(rainfall_series)
print("\nADF Statistic:", result[0])
print("p-value:", result[1])
if result[1] > 0.05:
    print("Series is non-stationary : differencing may be needed\n\n")
else:
    print("Series is stationary\n\n")


train_size = int(len(rainfall_series) * 0.8)
train, test = rainfall_series[:train_size], rainfall_series[train_size:]

# arima model
model = ARIMA(train, order=(4,0,2))
model_fit = model.fit()
forecast = model_fit.forecast(steps=len(test))

# plot 
plt.figure(figsize=(12, 5))
plt.plot(range(len(train)), train, label='Train')
plt.plot(range(len(train), len(train)+len(test)), test, label='Actual')
plt.plot(range(len(train), len(train)+len(test)), forecast, label='Forecast', linestyle='--')
plt.title('ARIMA Forecast vs Actual Rainfall')
plt.xlabel('Time Index')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.tight_layout()
plt.show()

# evaluation
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print("\nRMSE:", round(rmse, 2))
