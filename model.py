import pandas as pd
from sklearn.linear_model import LinearRegression
import psycopg2
import joblib

# ✅ Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost",
    database="airflow",
    user="airflow",
    password="airflow",
    port=5432
)

# ✅ Load data
df = pd.read_sql("SELECT * FROM rentals", conn)
conn.close()

# ✅ Features and target
X = df[["bedrooms", "price_per_room"]]
y = df["price"]

# ✅ Train model
model = LinearRegression()
model.fit(X, y)

# ✅ Save model
joblib.dump(model, "model.pkl")

print("✅ Model trained and saved successfully")
