from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# --- Load the DictVectorizer and model ---
with open("pipeline_v1.bin", "rb") as f:
    dv, model = pickle.load(f)

@app.post("/predict")
def predict(client: dict):
    # Convert dict to feature matrix
    X = dv.transform([client])
    
    # Compute the probability (if classifier)
    # Change to model.predict(X) if it’s a regressor
    try:
        y_pred = model.predict_proba(X)[0, 1]
    except AttributeError:
        y_pred = model.predict(X)[0]
    
    return {"prediction": float(y_pred)}
