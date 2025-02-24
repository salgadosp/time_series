# Time Series Forecasting with Python

This repository contains notebooks and datasets developed while studying the Udemy course *"Master Time Series Analysis and Forecasting with Python."*

It covers the following topics:

#### Exploratory Data Analysis (EDA) for Time Series:
- **Autocorrelation** – Measures how past values influence future values in a time series.
- **Seasonal Decomposition** – Breaks down a time series into trend, seasonality, and residual components.

#### Classical Time Series Models:
- **Exponential Smoothing** – Captures trend and seasonality by averaging past observations with decreasing weights.
- **(S)ARIMA(X)** – A statistical model that combines autoregression, differencing, and moving averages; supports seasonality (SARIMA) and exogenous variables (SARIMAX).  

#### Modern Forecasting Models:
- **Facebook Prophet** – Developed by Meta, this model is robust to missing data and handles seasonality and holidays well.  
- **LinkedIn Silverkite** – A forecasting model optimized for business use cases with interpretable components.

#### Deep Learning Models:
- **LSTM (Long Short-Term Memory)** – A deep learning approach using recurrent neural networks (RNNs) for time-dependent patterns.  
- **TFT (Temporal Fusion Transformers)** – A state-of-the-art deep learning model leveraging attention mechanisms for complex time series forecasting.  

## Libraries Used
This project uses the following Python libraries for time series analysis and forecasting:
- **numpy, pandas** – Data manipulation and numerical operations.
- **matplotlib, seaborn** – Data visualization.
- **scikit-learn** – Feature scaling, model evaluation, hyperparameter tuning, etc.
- **statsmodels** – Time series analysis.
- **pmdarima** – Automated ARIMA model selection and optimization.
- **prophet** – Meta's forecasting model.
- **greykite** – LinkedIn’s forecasting framework for business applications.
- **darts** – Deep learning and ensemble models for time series forecasting.
