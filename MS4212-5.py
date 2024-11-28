import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pandas as pd
from statsmodels.tsa.api import SARIMAX

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

# Fit SARIMAX model with constant term
model = SARIMAX(
    full_data['Fuel'],
    exog=full_data[['Fuel_Oil_Import', 'Gas_Oil_Diesel']],
    order=(1, 1, 1),
    seasonal_order=(0, 1, 1, 12),
    trend='c'  # Include constant term
)
model_fit = model.fit(disp=False)

# Extract residuals from the fitted model and remove the first outlier
residuals = model_fit.resid
residuals_cleaned = residuals.iloc[1:]  # Ignore the first residual

# Plot residuals
plt.figure(figsize=(12, 6))
plt.plot(residuals_cleaned, label="Residuals (Zoomed)", color="purple")
plt.axhline(0, linestyle="--", color="red", alpha=0.7)
plt.title("Residuals of the Model (Zoomed In)")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.legend()
plt.grid(True)
plt.show()

# Plot the ACF (Autocorrelation Function) of residuals
plt.figure(figsize=(10, 6))
plot_acf(residuals_cleaned, lags=40, title="Residuals ACF (Zoomed In)", zero=False)
plt.show()

# Plot the PACF (Partial Autocorrelation Function) of residuals
plt.figure(figsize=(10, 6))
plot_pacf(residuals_cleaned, lags=40, title="Residuals PACF (Zoomed In)", zero=False, method="ywm")
plt.show()
