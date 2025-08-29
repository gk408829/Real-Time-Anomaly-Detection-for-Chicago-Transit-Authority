# fetch_data.py

import os
import requests
from dotenv import load_dotenv
import json
import sqlite3
from datetime import datetime
import time

DB_PATH = os.path.join('data', 'cta_database.db')

def setup_database():
    """Creates the SQLite database and the train_positions table if they don't exist."""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table with relevant columns
    # We use "INTEGER" for timestamps and booleans for simplicity in SQLite
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS train_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetch_timestamp INTEGER NOT NULL,
            run_number TEXT NOT NULL,
            route_name TEXT NOT NULL,
            destination_name TEXT,
            next_station_name TEXT,
            arrival_time INTEGER,
            is_delayed INTEGER,
            latitude REAL,
            longitude REAL,
            heading INTEGER
        )
    ''')
    conn.commit()
    conn.close()
    print("Database setup complete.")

def fetch_cta_train_data():
    """Fetches real-time train position data from the CTA Train Tracker API."""
    load_dotenv()
    api_key = os.getenv("CTA_API_KEY")

    if not api_key:
        print("Error: CTA_API_KEY not found.")
        return None

    api_base_url = "http://lapi.transitchicago.com/api/1.0/ttpositions.aspx"
    params = {'key': api_key, 'rt': 'Red', 'outputType': 'JSON'}

    try:
        response = requests.get(api_base_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Data parsing error: {e}")
    return None

def process_and_save_data(data):
    """Processes the raw API data and inserts it into the SQLite database."""
    if not data:
        return

    try:
        # Get the parent 'route' object, which contains the route name
        route_object = data.get('ctatt', {}).get('route', [{}])[0]
        if not route_object:
            print("Could not find route data in the response.")
            return

        # --- FIX 1: Get the route name from the parent object ---
        # The route name is stored in the '@name' key
        route_name = route_object.get('@name')
        trains = route_object.get('train', [])

        if not trains:
            print("No active trains found in the data.")
            return
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        fetch_ts = int(time.time())
        
        records_to_insert = []
        for train in trains:
            arrival_ts = int(datetime.strptime(train.get('arrT'), '%Y-%m-%dT%H:%M:%S').timestamp()) if train.get('arrT') else None
            
            record = (
                fetch_ts,
                train.get('rn'),          # run_number (vehicle_id)
                route_name,               # --- FIX 2: Use the route_name variable we saved earlier ---
                train.get('destNm'),      # destination_name
                train.get('nextStaNm'),   # next_station_name
                arrival_ts,               # arrival_time
                int(train.get('isDly', 0)), # is_delayed (0 or 1)
                float(train.get('lat', 0.0)), # latitude
                float(train.get('lon', 0.0)), # longitude
                int(train.get('heading', 0))  # heading
            )
            records_to_insert.append(record)

        cursor.executemany('''
            INSERT INTO train_positions (
                fetch_timestamp, run_number, route_name, destination_name, 
                next_station_name, arrival_time, is_delayed, latitude, longitude, heading
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', records_to_insert)
        
        conn.commit()
        conn.close()
        
        print(f"Successfully saved data for {len(records_to_insert)} trains to the database.")

    except Exception as e:
        print(f"An error occurred during data processing: {e}")


if __name__ == "__main__":
    # First, ensure the database and table exist
    setup_database()

    # Run the script indefinitely
    while True:
        print(f"\n--- Running at {datetime.now()} ---")
        live_data = fetch_cta_train_data()
        process_and_save_data(live_data)
        
        # Wait for 60 seconds before the next run
        print("Waiting for 60 seconds...")
        time.sleep(60)
