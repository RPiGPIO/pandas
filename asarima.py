import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error

data = pd.read_csv("file.csv", parse_dates=['DATE'], index_col='DATE')

train = data[:int(len(data)*0.75)]
test = data[int(len(data)*0.75):]

auto_model = auto_arima(train['Value'], seasonal=True, m=12)

model = SARIMAX(train['Value'],
                order=auto_model.order,
                seasonal_order=auto_model.seasonal_order).fit()

forecast = model.forecast(len(test))

print("MSE:", mean_squared_error(test['Value'], forecast))
print("MAE:", mean_absolute_error(test['Value'], forecast))

# Plot
plt.figure(figsize=(10,5))
plt.plot(train.index, train['Value'], label='Train')
plt.plot(test.index, test['Value'], label='Actual')
plt.plot(test.index, forecast, label='Auto SARIMA', linestyle='--')
plt.legend()
plt.show()
