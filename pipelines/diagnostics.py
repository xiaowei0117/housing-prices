# diagnostics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

val = pd.read_csv("rf_model_outputs/val_predictions.csv")
rmse = np.sqrt(((val.y_true - val.y_pred) ** 2).mean())
print(f"RMSE = {rmse:,.2f}")

# prediction vs ture value
plt.figure(figsize=(5,5))
plt.scatter(val.y_true, val.y_pred, s=8)
plt.plot([val.y_true.min(), val.y_true.max()],
         [val.y_true.min(), val.y_true.max()], "--")
plt.xlabel("True"); plt.ylabel("Predicted"); plt.title("Pred vs True")
plt.tight_layout(); plt.show()

#
res = val.y_pred - val.y_true
plt.figure(figsize=(6,4))
plt.hist(res, bins=40)
plt.title("Residuals"); plt.xlabel("Pred - True"); plt.tight_layout(); plt.show()
