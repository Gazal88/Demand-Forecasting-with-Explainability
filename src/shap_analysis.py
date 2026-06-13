import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

model = joblib.load("../models/model.pkl")
X_test = pd.read_csv("../data/X_test.csv")

X_sample = X_test.sample(min(500, len(X_test)), random_state=42)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# --- Summary plot ---
plt.figure()
shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig("../outputs/shap_summary.png", dpi=120, bbox_inches="tight")
plt.close()

# --- Mean absolute SHAP value table ---
mean_abs_shap = pd.DataFrame({
    "feature": X_sample.columns,
    "mean_abs_shap": np.abs(shap_values).mean(axis=0)
}).sort_values("mean_abs_shap", ascending=False)

print("Feature importance (mean |SHAP value|):")
print(mean_abs_shap.to_string(index=False))

# --- Waterfall plot for one prediction ---
idx = 0
plt.figure()
shap.plots._waterfall.waterfall_legacy(
    explainer.expected_value,
    shap_values[idx],
    X_sample.iloc[idx],
    show=False
)
plt.tight_layout()
plt.savefig("../outputs/shap_waterfall_example.png", dpi=120, bbox_inches="tight")
plt.close()

print("\nSaved: shap_summary.png and shap_waterfall_example.png in outputs/")