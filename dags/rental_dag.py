from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import requests
import json
import psycopg2


# ✅ Task 1: Scrape data
def scrape_rental_data():
    listings = []

    for i in range(10):
        listing = {
            "title": f"Apartment {i}",
            "price": 1500 + i * 100,
            "location": "Toronto",
            "bedrooms": 1 + (i % 3),
        }
        listings.append(listing)

    print(f"✅ Scraped {len(listings)} listings")

    with open("/opt/airflow/dags/raw_rentals.json", "w") as f:
        json.dump(listings, f)


# ✅ Task 2: Process data
def process_data():
    with open("/opt/airflow/dags/raw_rentals.json", "r") as f:
        data = json.load(f)

    cleaned_data = []

    for item in data:
        cleaned_data.append({
            "title": item["title"],
            "price": float(item["price"]),
            "city": item["location"],
            "bedrooms": int(item["bedrooms"]),
            "price_per_room": float(item["price"]) / int(item["bedrooms"])
        })

    print("✅ Processed data")

    with open("/opt/airflow/dags/processed_rentals.json", "w") as f:
        json.dump(cleaned_data, f)


# ✅ Task 3: Load to DB
def load_to_postgres():
    conn = psycopg2.connect(
        host="postgres",
        database="airflow",
        user="airflow",
        password="airflow"
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rentals (
            id SERIAL PRIMARY KEY,
            title TEXT,
            price FLOAT,
            city TEXT,
            bedrooms INT,
            price_per_room FLOAT
        );
    """)

    with open("/opt/airflow/dags/processed_rentals.json", "r") as f:
        data = json.load(f)

    for item in data:
        cursor.execute("""
            INSERT INTO rentals (title, price, city, bedrooms, price_per_room)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            item["title"],
            item["price"],
            item["city"],
            item["bedrooms"],
            item["price_per_room"]
        ))

    conn.commit()
    cursor.close()
    conn.close()

    print("✅ Data loaded into PostgreSQL")


# ✅ DAG
with DAG(
    dag_id="rental_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@daily",
    catchup=False
) as dag:

    scrape_task = PythonOperator(
        task_id="scrape_data",
        python_callable=scrape_rental_data
    )

    process_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data
    )

    load_task = PythonOperator(
        task_id="load_to_postgres",
        python_callable=load_to_postgres
    )

    scrape_task >> process_task >> load_task