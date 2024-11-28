import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SARIMAX
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('C:/Users/jakethai2-c/Desktop/MS4212.csv')

# Clean up column names by removing unnecessary whitespace and characters
data.columns = data.columns.str.replace(r'[^A-Za-z0-9() ]+', '', regex=True).str.strip()

# Prepare columns by renaming and handling missing values
data['Fuel'] = data['Fuel'].astype(str).str.replace(" ", "").astype(float)
data['Fuel_Oil_Import'] = data['Fuel oil (Quantity Import)  (KL)'].astype(str).str.replace(" ", "").astype(float)
data['Gas_Oil_Diesel'] = data['Gas oil diesel oil and naphtha  (KL)'].astype(str).str.replace(" ", "").astype(float)

# Create a Date column for indexing
data['Date'] = pd.to_datetime(data[['Year', 'Month']].assign(Day=1))
data.set_index('Date', inplace=True)
data = data.asfreq('MS')  # Set frequency to monthly start

# Fill missing values in predictors
data['Fuel_Oil_Import'] = data['Fuel_Oil_Import'].ffill().bfill()
data['Gas_Oil_Diesel'] = data['Gas_Oil_Diesel'].ffill().bfill()

# Define time ranges for training, validation, and test sets
train_range = data.loc['2005':'2016', 'Fuel']
validation_range = data.loc['2017':'2018', 'Fuel']
test_range = data.loc['2019', 'Fuel']

# Define and fit the SARIMAX model with reduced complexity and specified ftol, xtol
model = SARIMAX(
    data.loc['2005':'2018', 'Fuel'],
    exog=data.loc['2005':'2018', ['Fuel_Oil_Import', 'Gas_Oil_Diesel']],
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 12)
)
model_fit = model.fit(disp=False, maxiter=1000, method='powell', ftol=1e-4, xtol=1e-4)

# Forecast on validation and test sets
forecast_validation = model_fit.predict(start=validation_range.index[0], end=validation_range.index[-1], exog=data.loc['2017':'2018', ['Fuel_Oil_Import', 'Gas_Oil_Diesel']])
forecast_test = model_fit.predict(start=test_range.index[0], end=test_range.index[-1], exog=data.loc['2019', ['Fuel_Oil_Import', 'Gas_Oil_Diesel']])

# Calculate RMSEs and mean values
train_fitted_values = model_fit.fittedvalues.loc[train_range.index]
stats_data = {
    'Data Set': ['Training', 'Validation', 'Test'],
    'RMSE': [
        np.sqrt(mean_squared_error(train_range, train_fitted_values)),  # RMSE for training (aligned)
        np.sqrt(mean_squared_error(validation_range, forecast_validation)),  # RMSE for validation
        np.sqrt(mean_squared_error(test_range, forecast_test))  # RMSE for test
    ],
    'Mean Actual': [train_range.mean(), validation_range.mean(), test_range.mean()],
    'Mean Predicted': [
        train_fitted_values.mean(),
        forecast_validation.mean(),
        forecast_test.mean()
    ]
}

# Create DataFrame with the statistics
stats_df = pd.DataFrame(stats_data)
print("Summary Table with Key Statistics:")
print(stats_df)

# Display AIC and BIC
print("\nModel AIC:", model_fit.aic)
print("Model BIC:", model_fit.bic)
print("\nModel Summary:")
print(model_fit.summary())
