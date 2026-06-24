from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

# ============================
# ✅ CREATE APP
# ============================
app = FastAPI()

# ============================
# ✅ ENABLE CORS
# ============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================
# ✅ LOAD MODEL SAFELY
# ============================
MODEL_PATH = "model.pkl"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("⚠️ Model not found — prediction will not work")

# ============================
# ✅ ROOT TEST
# ============================
@app.get("/")
def root():
    return {"message": "API is running ✅"}

# ============================
# ✅ RENTALS ENDPOINT
# ============================
@app.get("/rentals")
def get_rentals():
    try:
        file_path = "dags/processed_rentals.json"

        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Data file not found")

        df = pd.read_json(file_path)
        return df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================
# ✅ ML PREDICTION ENDPOINT
# ============================
@app.get("/predict")
def predict(bedrooms: int, price_per_room: float):

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        prediction = model.predict([[bedrooms, price_per_room]])
        return {"prediction": float(prediction[0])}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))