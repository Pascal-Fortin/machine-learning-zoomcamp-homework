import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------
df = pd.read_csv("GlobalTemperatures.csv")

# Convert date to datetime and sort
df['dt'] = pd.to_datetime(df['dt'])
df = df.sort_values('dt')

# Select relevant columns
df = df[['dt', 'LandAverageTemperature']].dropna()

# Rename for simplicity
df = df.rename(columns={'LandAverageTemperature': 'temp'})

# ---------------------------------------------------------------------
# 2. Feature engineering: add lag features (previous 12 months)
# ---------------------------------------------------------------------
for lag in range(1, 13):
    df[f'lag_{lag}'] = df['temp'].shift(lag)

# Drop initial rows with NaN values due to lagging
df = df.dropna().reset_index(drop=True)

# ---------------------------------------------------------------------
# 3. Train/validation/test split
# ---------------------------------------------------------------------
# Split by time (no shuffle for time-series)
n = len(df)
train_end = int(0.6 * n)
val_end = int(0.8 * n)

df_train = df.iloc[:train_end]
df_val = df.iloc[train_end:val_end]
df_test = df.iloc[val_end:]

X_train = df_train.drop(columns=['dt', 'temp'])
y_train = df_train['temp']
X_val = df_val.drop(columns=['dt', 'temp'])
y_val = df_val['temp']
X_test = df_test.drop(columns=['dt', 'temp'])
y_test = df_test['temp']

# ---------------------------------------------------------------------
# 4. Train three models
# ---------------------------------------------------------------------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

results = {}

# --- Decision Tree ---
dt = DecisionTreeRegressor(max_depth=5, random_state=1)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)
results['DecisionTree'] = rmse(y_val, y_pred_dt)

# --- Random Forest ---
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=1,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)
results['RandomForest'] = rmse(y_val, y_pred_rf)

# --- XGBoost ---
xgb = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=1,
    n_jobs=-1,
    objective='reg:squarederror'
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
y_pred_xgb = xgb.predict(X_val)
results['XGBoost'] = rmse(y_val, y_pred_xgb)

# ---------------------------------------------------------------------
# 5. Compare results
# ---------------------------------------------------------------------
print("Validation RMSE:")
for model, score in results.items():
    print(f"{model}: {score:.4f}")

# ---------------------------------------------------------------------
# 6. Evaluate best model on test data
# ---------------------------------------------------------------------
best_model_name = min(results, key=results.get)
print(f"\nBest model: {best_model_name}")

best_model = {'DecisionTree': dt, 'RandomForest': rf, 'XGBoost': xgb}[best_model_name]
y_pred_test = best_model.predict(X_test)
rmse_test = rmse(y_test, y_pred_test)
print(f"Test RMSE ({best_model_name}): {rmse_test:.4f}")

# ---------------------------------------------------------------------
# 7. Plot results
# ---------------------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(df_test['dt'], y_test, label="Actual", color='black')
plt.plot(df_test['dt'], y_pred_test, label=f"Predicted ({best_model_name})", color='red')
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.title("Actual vs Predicted Land Temperature (Test Set)")
plt.show()
