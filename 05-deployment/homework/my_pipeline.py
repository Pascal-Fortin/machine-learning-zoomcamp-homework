import pickle
import pandas as pd

# --- Step 1: Load your trained pipeline from a pickle file ---
with open("pipeline_v1.bin", "rb") as f:   # replace with your actual filename
    dv, pipeline = pickle.load(f)

# --- Step 2: Define the input record ---
record = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = dv.transform([record])
print(X)

# --- Step 3: Score the record ---
score = pipeline.predict_proba(X)
print(score)

# --- Step 4: Display the result ---
print("Predicted value:", score[0,1])
