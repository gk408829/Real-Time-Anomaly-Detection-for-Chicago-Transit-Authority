# fetch_data.py

import os
import requests
from dotenv import load_dotenv
import json
import sqlite3
from datetime import datetime
import time
import logging
import signal
import sys
import random

DB_PATH = os.path.join('data', 'cta_database.db')
POLLING_INTERVAL = int(os.getenv("POLLING_INTERVAL", 60))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)

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
    logging.info("Database setup complete.")

def validate_train_data(train):
    """Validate essential fields are present and reasonable."""
    required_fields = ['rn', 'lat', 'lon']
    for field in required_fields:
        if not train.get(field):
            return False
    
    # Basic coordinate validation for Chicago area
    try:
        lat, lon = float(train.get('lat', 0)), float(train.get('lon', 0))
        if not (41.6 <= lat <= 42.1 and -87.9 <= lon <= -87.5):
            return False
    except (ValueError, TypeError):
        return False
    
    return True

def fetch_cta_train_data(max_retries=3):
    """Fetches real-time train position data from the CTA Train Tracker API with backoff."""
    load_dotenv()
    api_key = os.getenv("CTA_API_KEY")

    if not api_key:
        logging.error("CTA_API_KEY not found in environment variables.")
        return None

    api_base_url = "http://lapi.transitchicago.com/api/1.0/ttpositions.aspx"
    # Fetch all CTA lines
    params = {'key': api_key, 'rt': 'Red,Blue,Brn,G,Org,P,Pink,Y', 'outputType': 'JSON'}

    for attempt in range(max_retries):
        try:
            response = requests.get(api_base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            logging.warning(f"API request error (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                logging.info(f"Retrying in {wait_time:.1f} seconds...")
                time.sleep(wait_time)
            else:
                logging.error("Max retries exceeded for API request")
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logging.error(f"Data parsing error: {e}")
            break
    return None

def log_data_quality(records_count, invalid_count):
    """Log data quality metrics."""
    logging.info(f"Processed {records_count} records, {invalid_count} invalid")
    if records_count > 0:
        quality_rate = (records_count - invalid_count) / records_count
        logging.info(f"Data quality rate: {quality_rate:.2%}")

def process_and_save_data(data):
    """Processes the raw API data and inserts it into the SQLite database."""
    if not data:
        return

    try:
        # Handle multiple routes in the response
        routes = data.get('ctatt', {}).get('route', [])
        if not isinstance(routes, list):
            routes = [routes]  # Single route case
        
        if not routes:
            logging.warning("Could not find route data in the response.")
            return

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        fetch_ts = int(time.time())
        
        all_records = []
        total_trains = 0
        invalid_trains = 0
        
        for route_object in routes:
            route_name = route_object.get('@name')
            trains = route_object.get('train', [])
            
            if not isinstance(trains, list):
                trains = [trains]  # Single train case
            
            for train in trains:
                total_trains += 1
                
                # Validate train data
                if not validate_train_data(train):
                    invalid_trains += 1
                    logging.debug(f"Invalid train data: {train}")
                    continue
                
                try:
                    arrival_ts = None
                    if train.get('arrT'):
                        arrival_ts = int(datetime.strptime(train.get('arrT'), '%Y-%m-%dT%H:%M:%S').timestamp())
                    
                    record = (
                        fetch_ts,
                        train.get('rn'),          # run_number (vehicle_id)
                        route_name,               # route_name
                        train.get('destNm'),      # destination_name
                        train.get('nextStaNm'),   # next_station_name
                        arrival_ts,               # arrival_time
                        int(train.get('isDly', 0)), # is_delayed (0 or 1)
                        float(train.get('lat', 0.0)), # latitude
                        float(train.get('lon', 0.0)), # longitude
                        int(train.get('heading', 0))  # heading
                    )
                    all_records.append(record)
                    
                except (ValueError, TypeError) as e:
                    invalid_trains += 1
                    logging.debug(f"Error processing train {train.get('rn')}: {e}")

        if all_records:
            cursor.executemany('''
                INSERT INTO train_positions (
                    fetch_timestamp, run_number, route_name, destination_name, 
                    next_station_name, arrival_time, is_delayed, latitude, longitude, heading
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', all_records)
            
            conn.commit()
            logging.info(f"Successfully saved data for {len(all_records)} trains to the database.")
        else:
            logging.warning("No valid train records to save.")
        
        conn.close()
        log_data_quality(total_trains, invalid_trains)

    except Exception as e:
        logging.error(f"An error occurred during data processing: {e}")


def signal_handler(sig, frame):
    """Handle graceful shutdown."""
    logging.info('Gracefully shutting down...')
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # First, ensure the database and table exist
    setup_database()
    
    logging.info(f"Starting data collection with {POLLING_INTERVAL}s intervals")
    logging.info("Press Ctrl+C to stop gracefully")

    # Run the script indefinitely
    while True:
        logging.info(f"--- Fetching data at {datetime.now()} ---")
        live_data = fetch_cta_train_data()
        process_and_save_data(live_data)
        
        # Wait for the configured interval
        logging.info(f"Waiting for {POLLING_INTERVAL} seconds...")
        time.sleep(POLLING_INTERVAL)
