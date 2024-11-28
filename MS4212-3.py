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

# Ensure the dataset covers 2005–2019 (180 months)
full_data = data.loc['2005':'2019']

# Check if exogenous variables have the correct shape (180, 2)
assert full_data[['Fuel_Oil_Import', 'Gas_Oil_Diesel']].shape[0] == 180, "Exogenous data does not cover 180 months."

# Fit SARIMAX model with constant term
model = SARIMAX(
    full_data['Fuel'],
    exog=full_data[['Fuel_Oil_Import', 'Gas_Oil_Diesel']],
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    trend='c'  # Include constant term
)
model_fit = model.fit(disp=False)

# Predict fuel consumption for the entire period (180 months)
forecast_full = model_fit.predict(
    start=full_data.index[0],
    end=full_data.index[-1],
    exog=full_data[['Fuel_Oil_Import', 'Gas_Oil_Diesel']]
)

# Plot actual vs. predicted values
plt.figure(figsize=(12, 6))
plt.plot(full_data['Fuel'], label="Actual Fuel Consumption", color="blue")
plt.plot(forecast_full, label="Predicted Fuel Consumption", color="orange", linestyle="--")
plt.title("Fuel Consumption: Actual vs. Predicted (2005–2019)")
plt.xlabel("Date")
plt.ylabel("Fuel Consumption")
plt.legend()
plt.grid(True)
plt.show()

# Calculate RMSE
rmse_full = np.sqrt(mean_squared_error(full_data['Fuel'], forecast_full))
print(f"RMSE for Full Prediction (2005–2019): {rmse_full}")

# Model summary
print("\nModel Summary:")
print(model_fit.summary())
