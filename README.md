# Rainfall-prediction-using-ARIMA
---

## Overview
This project implements a rainfall prediction model using the ARIMA (AutoRegressive Integrated Moving Average) time series approach. The dataset contains daily rainfall measurements, which are aggregated into monthly precipitation values for modeling. The model is trained on 80% of the data and evaluated on the remaining 20%.

## Dataset
- **Input Data**: `Rainfall_data.csv`
- **Columns**:
  - `Year`: Year of measurement
  - `Month`: Month of measurement
  - `Day`: Day of measurement
  - `Precipitation`: Daily recorded rainfall (mm)

## Preprocessing Steps
1. Convert `Year`, `Month`, and `Day` columns into a `Date` column.
2. Aggregate daily precipitation values into monthly totals.
3. Convert monthly data into a time series format.
4. Perform stationarity check using the Augmented Dickey-Fuller (ADF) test.
5. If the series is non-stationary, apply differencing.

## Model
- **Algorithm**: ARIMA
- **Order**: (4,0,2)
- **Evaluation Metric**: Root Mean Squared Error (RMSE)

## ARIMA Model Description

### Uses and Working of Algorithms

The Autoregressive Integrated Moving Average (ARIMA) model is a popular choice for time series forecasting, particularly when dealing with non-seasonal data that exhibits a certain degree of autocorrelation. Here's how ARIMA works and its relevance to rainfall prediction:

1. **AutoRegressive (AR) Component**: This component models the relationship between an observation and a number of lagged observations (i.e., how previous time points influence the current one). The order of the AR component, denoted as `p`, specifies the number of lag observations included in the model.

2. **Integrated (I) Component**: The integration component accounts for differencing, which transforms the original time series into a stationary series. A stationary series has constant mean and variance over time. The order of differencing, denoted as `d`, determines the number of times the differencing operation is applied to achieve stationarity.

3. **Moving Average (MA) Component**: This component models the dependency between an observation and a residual error from a moving average model applied to lagged observations. The order of the MA component, denoted as `q`, specifies the number of lagged forecast errors in the prediction equation.

### Parameters Relevance

- **Order (p, d, q)**: The ARIMA model is defined by its order parameters `(p, d, q)`. In this project, the ARIMA order is specified as `(4, 0, 2)`:
  - `p = 4`: The number of lag observations included in the model.
  - `d = 0`: No differencing applied, indicating the data was already stationary after preprocessing steps.
  - `q = 2`: The size of the moving average window.

### Model Evaluation Metric

- **Evaluation Metric**: Root Mean Squared Error (RMSE) is used to assess the performance of the ARIMA model in predicting monthly rainfall totals. RMSE measures the average magnitude of the prediction errors, with lower values indicating better model performance.



