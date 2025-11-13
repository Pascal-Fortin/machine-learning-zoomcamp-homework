# Concrete Compressive Strength Regression Pipeline

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# 1️⃣ Load the dataset (CSV file)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
df = pd.read_excel(url)

# Rename columns for easier reference
df.columns = [
    "cement",
    "blast_furnace_slag",
    "fly_ash",
    "water",
    "superplasticizer",
    "coarse_aggregate",
    "fine_aggregate",
    "age",
    "strength"
]

print(df.head())

# 2️⃣ Split the data
X = df.drop("strength", axis=1)
y = df["strength"]

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.25, random_state=1)  # 0.25 x 0.8 = 0.2

print(f"Train shape: {X_train.shape}, Validation shape: {X_val.shape}, Test shape: {X_test.shape}")

# 3️⃣ Decision Tree
dt = DecisionTreeRegressor(max_depth=5, random_state=1)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)
rmse_dt = mean_squared_error(y_val, y_pred_dt, squared=False)
print(f"Decision Tree RMSE: {rmse_dt:.3f}")

# 4️⃣ Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
rmse_rf = mean_squared_error(y_val, y_pred_rf, squared=False)
print(f"Random Forest RMSE: {rmse_rf:.3f}")

# 5️⃣ XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

xgb_params = {
    "eta": 0.1,
    "max_depth": 6,
    "objective": "reg:squarederror",
    "seed": 1,
    "nthread": 8,
    "verbosity": 0
}

watchlist = [(dtrain, "train"), (dval, "val")]
xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=200, evals=watchlist, early_stopping_rounds=20, verbose_eval=False)

y_pred_xgb = xgb_model.predict(dval)
rmse_xgb = mean_squared_error(y_val, y_pred_xgb, squared=False)
print(f"XGBoost RMSE: {rmse_xgb:.3f}")

# 6️⃣ Compare performance
models = ["Decision Tree", "Random Forest", "XGBoost"]
rmses = [rmse_dt, rmse_rf, rmse_xgb]

plt.figure(figsize=(7,4))
plt.bar(models, rmses, color=["#5DADE2", "#58D68D", "#F4D03F"])
plt.ylabel("Validation RMSE")
plt.title("Model Performance on Concrete Strength Dataset")
plt.show()

# 7️⃣ Check feature importances (Random Forest example)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances)
