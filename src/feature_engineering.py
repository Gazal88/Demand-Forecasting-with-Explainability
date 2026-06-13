import pandas as pd
import numpy as np
import holidays

df = pd.read_csv("../data/train.csv", parse_dates=["date"])

# Sort so lag/rolling features are computed correctly within each store-item group
df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)

# --- Date-based features ---
df["day_of_week"] = df["date"].dt.dayofweek      # 0=Monday, 6=Sunday
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["day_of_year"] = df["date"].dt.dayofyear

# --- Holiday feature (US holidays, since dataset is US retail) ---
us_holidays = holidays.US()
df["is_holiday"] = df["date"].isin(us_holidays).astype(int)

# --- Lag features (per store-item group) ---
for lag in [7, 28]:
    df[f"sales_lag_{lag}"] = df.groupby(["store", "item"])["sales"].shift(lag)

# --- Rolling average features ---
for window in [7, 28]:
    df[f"sales_roll_mean_{window}"] = (
        df.groupby(["store", "item"])["sales"]
        .shift(1)
        .rolling(window)
        .mean()
        .reset_index(drop=True)
    )


df = df.dropna().reset_index(drop=True)

print(f"Shape after feature engineering: {df.shape}")
print(df.columns.tolist())
print(df.head())

df.to_csv("../data/features.csv", index=False)