# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```python
# Import the necessary packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("/content/cardekho.csv")

# Display dataset info to understand the data
print("Dataset columns:", data.columns.tolist())

# Assuming the dataset has a numeric column - using the first numeric one
numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
target_variable = numeric_columns[0] if numeric_columns else None

if target_variable:
    print(f"Using '{target_variable}' as target variable")

    # ARIMA Model function (same as in PDF)
    def arima_model(data, target_variable, order):
        train_size = int(len(data) * 0.8)
        train_data, test_data = data[:train_size], data[train_size:]

        model = ARIMA(train_data[target_variable], order=order)
        fitted_model = model.fit()

        forecast = fitted_model.forecast(steps=len(test_data))

        rmse = np.sqrt(mean_squared_error(test_data[target_variable], forecast))

        plt.figure(figsize=(10, 6))
        plt.plot(train_data.index, train_data[target_variable], label='Training Data')
        plt.plot(test_data.index, test_data[target_variable], label='Testing Data')
        plt.plot(test_data.index, forecast, label='Forecasted Data')
        plt.xlabel('Date')
        plt.ylabel(target_variable)
        plt.title('ARIMA Forecasting for ' + target_variable)
        plt.legend()
        plt.show()

        print("Root Mean Squared Error (RMSE):", rmse)

    # Run ARIMA model with parameters (same as in PDF)
    arima_model(data, target_variable, order=(2,1,2))

    print("\nRESULT:")
    print("Thus the program run successfully based on the ARIMA model using python.")
else:
    print("No numeric columns found in the dataset")
```

### OUTPUT:
<img width="1055" height="527" alt="image" src="https://github.com/user-attachments/assets/c714a0e9-050a-4ff4-8f2e-9c6b2b1b0fc2" />



### RESULT:
Thus the program run successfully based on the ARIMA model using python.
