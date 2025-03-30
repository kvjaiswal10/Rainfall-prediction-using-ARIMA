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


