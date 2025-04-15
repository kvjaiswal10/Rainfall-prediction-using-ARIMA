import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

# data
df = pd.read_csv(r"C:\Users\kvjai\ML PROJECTS\Rainfall prediction - ARIMA\data\Rainfall_data.csv")

# merge year, month and day columns into datetime obj and sort in chronological order
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
df = df.sort_values(by='Date')

# aggreagate to month wise data, converts date to year-month format
df['YearMonth'] = df['Date'].dt.to_period('M')
# group data by month and sum precipitation to get monthly total rainfall
monthly_rainfall = df.groupby('YearMonth')['Precipitation'].sum().reset_index()
monthly_rainfall['YearMonth'] = monthly_rainfall['YearMonth'].astype(str)

# Extract precipitation values
rainfall_series = monthly_rainfall['Precipitation']

# stationarity check to see if stats like mean and variance dont change over time
result = adfuller(rainfall_series)
print("\nADF Statistic:", result[0])
print("p-value:", result[1])
# p>0.05 means series is non stationary and may need differencing as we need stationary data for ARIMA
if result[1] > 0.05:
    print("Series is non-stationary : differencing may be needed\n\n")
    print(result[1])
else:
    print("Series is stationary\n\n")


train_size = int(len(rainfall_series) * 0.8)
train, test = rainfall_series[:train_size], rainfall_series[train_size:]

# -------------------------------
# Fit SARIMA model (seasonal order assumed monthly)
# order=(p,d,q), seasonal_order=(P,D,Q,s) where s=12 for yearly seasonality
# -------------------------------
model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit(disp=False)

# forecast
forecast = model_fit.forecast(steps=len(test))

# -------------------------------
# Plot Forecast vs Actual
# -------------------------------
plt.figure(figsize=(12, 5))
plt.plot(range(len(train)), train, label='Train')
plt.plot(range(len(train), len(train)+len(test)), test, label='Actual')
plt.plot(range(len(train), len(train)+len(test)), forecast, label='Forecast', linestyle='--')
plt.title('SARIMA Forecast vs Actual Rainfall')
plt.xlabel('Time Index')
plt.ylabel('Precipitation (mm)')
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------
# Evaluation
# -------------------------------
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print("\nRMSE:", round(rmse, 2))


'''
P	Seasonal AR	Same as p, but for seasonal lags (e.g., 12 months back).
D	Seasonal Differencing	Same as d, but applied on the seasonal cycle.
Q	Seasonal MA	Same as q, but on seasonal errors.
s	Seasonal Period (e.g., s=12)	The length of the seasonal cycle (12 for monthly data).

Example: SARIMA(1, 1, 1) x (1, 1, 1, 12)
Uses 1 lag of past value (p=1)

Applies 1 differencing (d=1)

Uses 1 lag of forecast error (q=1)

Also includes seasonal lags from 12 months back

Seasonality repeats every 12 months (s=12)

Applies seasonal differencing once (D=1)

------------------------------------------------------------------------------------------------------

AR (AutoRegressive)
Model uses the past values of the series to predict future values.

If p = 2, the model uses the last 2 values.

I (Integrated)
Refers to differencing the data to make it stationary (constant mean/variance over time).

If d = 1, it subtracts the previous value:
y_t - y_(t-1)

MA (Moving Average)
Model uses the past errors (residuals) to make the prediction.

If q = 2, it uses the last 2 error terms.


'''
