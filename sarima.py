import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("file.csv", parse_dates=['DATE'], index_col='DATE')

train = data[:int(len(data)*0.75)]
test = data[int(len(data)*0.75):]

model = SARIMAX(train['Value'], order=(1,1,1), seasonal_order=(1,1,0,12)).fit()

forecast = model.forecast(len(test))

print("MSE:", mean_squared_error(test['Value'], forecast))
print("MAE:", mean_absolute_error(test['Value'], forecast))

# Plot
plt.figure(figsize=(10,5))
plt.plot(train.index, train['Value'], label='Train')
plt.plot(test.index, test['Value'], label='Actual')
plt.plot(test.index, forecast, label='SARIMA', linestyle='--')
plt.legend()
plt.show()
