import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error


data = pd.read_csv("file.csv")


data['DATE'] = pd.to_datetime(data['DATE'])


data['Value'] = data['Value'].fillna(method='ffill')


data.set_index('DATE', inplace=True)

train = data[:int(len(data)*0.75)]
test = data[int(len(data)*0.75):]


# =========================
# 1) ARIMA
# =========================
# model = ARIMA(train['Value'], order=(1,1,1)).fit()
# forecast = model.forecast(len(test))
# model_name = "ARIMA"

# =========================
# 2) AUTO ARIMA
# =========================
# auto_model = auto_arima(train['Value'], seasonal=False)
# model = ARIMA(train['Value'], order=auto_model.order).fit()
# forecast = model.forecast(len(test))
# model_name = "AUTO ARIMA"

# =========================
# 3) SARIMA
# =========================
# model = SARIMAX(
#     train['Value'],
#     order=(1,1,1),
#     seasonal_order=(1,1,0,12)
# ).fit()
# forecast = model.forecast(len(test))
# model_name = "SARIMA"

# =========================
# 4) AUTO SARIMA
# =========================
auto_model = auto_arima(train['Value'], seasonal=True, m=12)
model = SARIMAX(
    train['Value'],
    order=auto_model.order,
    seasonal_order=auto_model.seasonal_order
).fit()
forecast = model.forecast(len(test))
model_name = "AUTO SARIMA"

# =========================
# Evaluation
# =========================
print("MSE:", mean_squared_error(test['Value'], forecast))
print("MAE:", mean_absolute_error(test['Value'], forecast))

# =========================
# Plot
# =========================
plt.figure(figsize=(10,5))
plt.plot(train.index, train['Value'], label='Train')
plt.plot(test.index, test['Value'], label='Actual')
plt.plot(test.index, forecast, label=model_name, linestyle='--')
plt.legend()
plt.show()
