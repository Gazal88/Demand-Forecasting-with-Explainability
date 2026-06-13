import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

test_meta = pd.read_csv("../data/test_meta.csv", parse_dates=["date"])
y_test = pd.read_csv("../data/y_test.csv")
preds = pd.read_csv("../data/preds.csv")

test_meta["actual"] = y_test.values
test_meta["predicted"] = preds.values

# Pick the first available store-item combination
first_store = test_meta["store"].iloc[0]
first_item = test_meta["item"].iloc[0]

sample = test_meta[(test_meta["store"] == first_store) & (test_meta["item"] == first_item)].sort_values("date")

plt.figure(figsize=(10, 5))
plt.plot(sample["date"], sample["actual"], label="Actual", marker="o", markersize=3)
plt.plot(sample["date"], sample["predicted"], label="Predicted", marker="x", markersize=3)
plt.title(f"Demand Forecast: Store {first_store}, Item {first_item}")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../outputs/pred_vs_actual.png", dpi=120)
plt.close()
print("Saved pred_vs_actual.png")