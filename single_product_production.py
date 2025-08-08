import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
#load data
df=pd.read_csv('data/sales_data.csv')
#filter sales data for a single product
product_id = "FOO_09"  # Change this to the desired product ID
df_product= df[df['product_id'] == product_id]
#Grou[by date to get daily sales]
df_grouped=df_product.groupby("date")["units_sold"].sum().reset_index()
print(df_grouped.head())
df_grouped.columns=["ds","y"]#rename for prophet
#train the model
model=Prophet(daily_seasonality=True)
model.fit(df_grouped)
#forecast
future=model.make_future_dataframe(periods=30)  # Forecast for the next 30 days
forecast=model.predict(future)
#plot forecast
fig=model.plot(forecast)
plt.title(f"Demand Forecast for{product_id}")
plt.show()