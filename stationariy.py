import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

# Read the CSV file into a DataFrame
file_path = 'QQQ.csv'
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime and set it as the index
df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date')

# Apply logarithmic transformation
df['Close_logs'] = np.log(df['Adj Close'])

# Calculate first differences
df["Close_diff"] = df['Close_logs'].diff()
df = df.dropna()

# Drop the specified columns
columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close', 'Close_logs']
df = df.drop(columns=columns_to_drop)

# Calculate ACF values and confidence intervals
lags = 30
acf_values, conf_intervals = acf(df['Close_diff'], nlags=lags, alpha=0.05)

# Calculate standard errors
std_errors = conf_intervals[:, 1] - acf_values

# Plot ACF as a line graph with error bars
plt.errorbar(range(0, lags + 1), acf_values, yerr=std_errors, fmt='o-', label='ACF values with 95% Confidence Interval')

# Add a horizontal axis at y=0
plt.axhline(y=0)

plt.title("Autocorrelation Function (ACF) for Adjusted Close Price",fontsize=12)
plt.xlabel("Lags",fontsize=12)
plt.ylabel("ACF",fontsize=12)
plt.legend()
plt.show()
