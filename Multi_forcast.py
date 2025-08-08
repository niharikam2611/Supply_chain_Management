import os
import pandas as pd
from prophet import Prophet
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
os.makedirs("Multiforecast", exist_ok=True)
df = pd.read_csv("data/sales_data.csv")
df["date"] = pd.to_datetime(df["date"])
unique_products = df["product_id"].unique()
print(f"Generating forecasts for {len(unique_products)} products...\n")
for pid in tqdm(unique_products):
    df_product = df[df["product_id"] == pid]
    df_grouped = df_product.groupby("date")["units_sold"].sum().reset_index()
    df_grouped.columns = ["ds", "y"]
    df_grouped["ds"] = pd.to_datetime(df_grouped["ds"])
    if len(df_grouped) < 30:
        continue
    try:
        model = Prophet(daily_seasonality=True)
        model.fit(df_grouped)
        future = model.make_future_dataframe(periods=30)
        forecast = model.predict(future)
        forecast["product_id"] = pid
        forecast.to_csv(f"Multiforecast/forecast_{pid}.csv", index=False)
    except Exception as e:
        print(f"Skipped {pid} due to error: {e}")
        continue
print("\n All forecassts generated and saved in the 'Multiforecast' directory.")