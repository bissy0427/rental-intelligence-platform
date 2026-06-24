import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# ✅ Sample training data (same as your app data)
data = {
    "bedrooms": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    "price_per_room": [1500, 800, 566, 1800, 950, 666, 2100, 1100, 766, 2400],
    "price": [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400]
}

df = pd.DataFrame(data)

X = df[["bedrooms", "price_per_room"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

# ✅ SAVE MODEL
joblib.dump(model, "model.pkl")

print("✅ Model created successfully")
