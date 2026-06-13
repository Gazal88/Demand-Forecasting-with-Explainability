import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import joblib

df = pd.read_csv("../data/features.csv", parse_dates=["date"])


split_date = df["date"].max() - pd.DateOffset(months=3)
train = df[df["date"] < split_date]
test = df[df["date"] >= split_date]

feature_cols = [
    "store", "item", "is_holiday",
    "day_of_week", "month", "year", "is_weekend", "day_of_year",
    "sales_lag_7", "sales_lag_28", "sales_roll_mean_7", "sales_roll_mean_28"
]
target_col = "sales"

X_train, y_train = train[feature_cols], train[target_col]
X_test, y_test = test[feature_cols], test[target_col]

print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")
print(f"Train date range: {train['date'].min()} to {train['date'].max()}")
print(f"Test date range: {test['date'].min()} to {test['date'].max()}")

model = lgb.LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(
    X_train, y_train,
    categorical_feature=["store", "item"]
)

preds = model.predict(X_test)
preds = np.clip(preds, 0, None)

rmse = np.sqrt(mean_squared_error(y_test, preds))
mape = mean_absolute_percentage_error(y_test, preds) * 100

print(f"\nRMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")

joblib.dump(model, "../models/model.pkl")
X_test.to_csv("../data/X_test.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)
test[["date", "store", "item"]].to_csv("../data/test_meta.csv", index=False)
pd.Series(preds).to_csv("../data/preds.csv", index=False)