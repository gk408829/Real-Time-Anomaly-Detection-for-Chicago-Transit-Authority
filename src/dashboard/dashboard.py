"""
Streamlit Dashboard for CTA Train Anomaly Detection

This dashboard provides an interactive interface for:
- Testing the anomaly detection API
- Visualizing train data on Chicago map
- Exploring model performance
- Real-time anomaly detection demo
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import json
import sqlite3
from math import radians, cos, sin, asin, sqrt

# Page configuration
st.set_page_config(
    page_title="CTA Train Anomaly Detection",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# CTA Line Colors (official colors) - mapped to actual route names in data
CTA_LINE_COLORS = {
    'red': '#C60C30',      # Red Line
    'blue': '#00A1DE',     # Blue Line  
    'brn': '#62361B',      # Brown Line (abbreviated as 'brn' in data)
    'g': '#009639',        # Green Line (abbreviated as 'g' in data)
    'org': '#F9461C',      # Orange Line (abbreviated as 'org' in data)
    'p': '#522398',        # Purple Line (abbreviated as 'p' in data)
    'pink': '#E27EA6',     # Pink Line
    'y': '#F9E300'         # Yellow Line (abbreviated as 'y' in data)
}

# Cache functions for better performance
@st.cache_data
def load_train_data():
    """Load recent train data from database"""
    try:
        DB_PATH = "data/cta_database.db"
        conn = sqlite3.connect(DB_PATH)
        
        # Get recent data (last 1000 records)
        query = """
        SELECT * FROM train_positions 
        ORDER BY fetch_timestamp DESC 
        LIMIT 1000
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        # Add datetime column
        df['datetime'] = pd.to_datetime(df['fetch_timestamp'], unit='s')
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def get_api_info():
    """Get API model information"""
    try:
        response = requests.get(f"{API_BASE_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

@st.cache_data
def get_supported_routes():
    """Get supported CTA routes from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/routes", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return ['red', 'blue', 'brn', 'g', 'org', 'p', 'pink', 'y']
    except:
        return ['red', 'blue', 'brn', 'g', 'org', 'p', 'pink', 'y']

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_anomaly(train_data):
    """Call API to predict anomaly"""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=train_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def main():
    """Main dashboard application"""
    
    # Title and header
    st.title("CTA Train Anomaly Detection Dashboard")
    st.markdown("Real-time anomaly detection for Chicago Transit Authority trains")
    
    # Check API status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("API is running and healthy")
    else:
        st.error("API is not responding. Please start the API server first.")
        st.code("python start_api.py")
        return
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Overview", "Live Detection", "Data Explorer", "Model Performance"]
    )
    
    if page == "Overview":
        show_overview()
    elif page == "Live Detection":
        show_live_detection()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Model Performance":
        show_model_performance()

def show_overview():
    """Show project overview and system status"""
    st.header("Project Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Project Goals")
        st.markdown("""
        - **Real-time Monitoring**: Detect anomalies in CTA train behavior
        - **Early Warning System**: Alert operators to potential issues
        - **Data-Driven Insights**: Understand transit patterns and bottlenecks
        - **Passenger Benefits**: Improve reliability and reduce delays
        """)
        
        st.subheader("Technical Stack")
        st.markdown("""
        - **Data Collection**: CTA Train Tracker API
        - **Storage**: SQLite database
        - **ML Models**: LightGBM, Random Forest, Isolation Forest
        - **API**: FastAPI with real-time predictions
        - **Dashboard**: Streamlit with interactive visualizations
        """)
    
    with col2:
        st.subheader("System Status")
        
        # API status
        api_info = get_api_info()
        if api_info:
            st.metric("Model Type", api_info['model_type'])
            st.metric("AUC-ROC Score", f"{api_info['performance']['auc_roc']:.3f}")
            st.metric("Precision", f"{api_info['performance']['precision']:.3f}")
            st.metric("Recall", f"{api_info['performance']['recall']:.3f}")
        
        # Data status
        df = load_train_data()
        if not df.empty:
            st.metric("Recent Records", f"{len(df):,}")
            st.metric("Routes Covered", df['route_name'].nunique())
            st.metric("Latest Update", df['datetime'].max().strftime("%Y-%m-%d %H:%M"))

def show_live_detection():
    """Interactive anomaly detection interface"""
    st.header("Live Anomaly Detection")
    st.markdown("Test the anomaly detection model with custom train data")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Train Information")
            route_name = st.selectbox("Route", get_supported_routes())
            speed_kmh = st.slider("Speed (km/h)", 0.0, 100.0, 30.0, 0.1)
            heading = st.slider("Heading (degrees)", 0, 360, 180, 1)
            
        with col2:
            st.subheader("Location")
            latitude = st.slider("Latitude", 41.6, 42.1, 41.8781, 0.0001)
            longitude = st.slider("Longitude", -87.9, -87.5, -87.6298, 0.0001)
            is_delayed = st.checkbox("Train is delayed")
            
        with col3:
            st.subheader("Time Context")
            hour_of_day = st.slider("Hour of Day", 0, 23, 14, 1)
            day_of_week = st.selectbox("Day of Week", {
                0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
                4: "Friday", 5: "Saturday", 6: "Sunday"
            })
            is_weekend = day_of_week >= 5
            is_rush_hour = hour_of_day in [7, 8, 9, 17, 18, 19]
            
            st.info(f"Weekend: {'Yes' if is_weekend else 'No'}")
            st.info(f"Rush Hour: {'Yes' if is_rush_hour else 'No'}")
        
        submitted = st.form_submit_button("🔮 Predict Anomaly", use_container_width=True)
        
        if submitted:
            # Prepare data for API
            train_data = {
                "speed_kmh": speed_kmh,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "is_delayed": int(is_delayed),
                "heading": heading,
                "latitude": latitude,
                "longitude": longitude,
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush_hour,
                "route_name": route_name
            }
            
            # Make prediction
            with st.spinner("Making prediction..."):
                result = predict_anomaly(train_data)
            
            if result:
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result['is_anomaly']:
                        st.error("ANOMALY DETECTED")
                    else:
                        st.success("NORMAL BEHAVIOR")
                
                with col2:
                    st.metric("Anomaly Probability", f"{result['anomaly_probability']:.1%}")
                
                with col3:
                    st.metric("Model Confidence", f"{result['confidence_score']:.1%}")
                
                # Show detailed results
                with st.expander("Detailed Results"):
                    st.json(result)

def show_data_explorer():
    """Explore the collected train data"""
    st.header("Data Explorer")
    st.markdown("Explore patterns in the collected CTA train data")
    
    # Load data
    df = load_train_data()
    if df.empty:
        st.warning("No data available. Make sure the data collection script is running.")
        return
    
    # Data overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Unique Routes", df['route_name'].nunique())
    with col3:
        st.metric("Unique Trains", df['run_number'].nunique())
    with col4:
        st.metric("Delay Rate", f"{df['is_delayed'].mean():.1%}")
    
    # Route distribution
    st.subheader("Route Distribution")
    route_counts = df['route_name'].value_counts()
    
    # Create color mapping for the bar chart
    colors = [CTA_LINE_COLORS.get(route, '#888888') for route in route_counts.index]
    
    fig = px.bar(x=route_counts.index, y=route_counts.values, 
                 title="Records by CTA Route",
                 labels={'x': 'Route', 'y': 'Number of Records'},
                 color=route_counts.index,
                 color_discrete_map=CTA_LINE_COLORS)
    st.plotly_chart(fig, use_container_width=True)
    
    # Time patterns
    st.subheader("⏰ Temporal Patterns")
    df['hour'] = df['datetime'].dt.hour
    hourly_counts = df.groupby('hour').size()
    
    fig = px.line(x=hourly_counts.index, y=hourly_counts.values,
                  title="Train Activity by Hour of Day",
                  labels={'x': 'Hour of Day', 'y': 'Number of Records'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive map
    st.subheader("Train Positions Map")
    
    # Sample data for map (limit to 1000 points for performance)
    map_data = df.sample(min(1000, len(df)))
    
    fig = px.scatter_mapbox(
        map_data,
        lat="latitude",
        lon="longitude",
        color="route_name",
        hover_data=["run_number", "destination_name", "is_delayed"],
        title="Recent CTA Train Positions",
        mapbox_style="open-street-map",
        zoom=10,
        center={"lat": 41.8781, "lon": -87.6298},
        color_discrete_map=CTA_LINE_COLORS
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data table
    st.subheader("Raw Data Sample")
    st.dataframe(df.head(100), use_container_width=True)

def show_model_performance():
    """Show model performance metrics and analysis"""
    st.header("Model Performance")
    st.markdown("Analysis of the anomaly detection model performance")
    
    # Get model info
    api_info = get_api_info()
    if not api_info:
        st.error("Could not retrieve model information from API")
        return
    
    # Performance metrics
    st.subheader("Model Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    perf = api_info['performance']
    with col1:
        st.metric("AUC-ROC", f"{perf['auc_roc']:.3f}")
    with col2:
        st.metric("Precision", f"{perf['precision']:.3f}")
    with col3:
        st.metric("Recall", f"{perf['recall']:.3f}")
    with col4:
        st.metric("F1-Score", f"{perf['f1_score']:.3f}")
    
    # Model details
    st.subheader("Model Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Model Type**: {api_info['model_type']}")
        st.info(f"**Features Used**: {len(api_info['features'])}")
        
        # Show features
        st.markdown("**Feature List:**")
        for i, feature in enumerate(api_info['features'], 1):
            st.markdown(f"{i}. `{feature}`")
    
    with col2:
        # Performance interpretation
        st.markdown("**Performance Interpretation:**")
        
        auc_roc = perf['auc_roc']
        if auc_roc > 0.9:
            st.success("Excellent performance (AUC-ROC > 0.9)")
        elif auc_roc > 0.8:
            st.success("Good performance (AUC-ROC > 0.8)")
        elif auc_roc > 0.7:
            st.warning("Fair performance (AUC-ROC > 0.7)")
        else:
            st.error("Poor performance (AUC-ROC < 0.7)")
        
        precision = perf['precision']
        recall = perf['recall']
        
        st.markdown(f"""
        - **Precision ({precision:.1%})**: Of predicted anomalies, {precision:.1%} are correct
        - **Recall ({recall:.1%})**: Model detects {recall:.1%} of actual anomalies
        - **Trade-off**: {'High precision, low recall' if precision > recall else 'High recall, low precision'}
        """)
    
    # Confusion matrix visualization
    st.subheader("Model Behavior Analysis")
    
    # Create sample predictions for different scenarios
    scenarios = [
        {"name": "Normal Rush Hour", "speed": 25, "hour": 8, "delayed": False, "expected": "Normal"},
        {"name": "Very Fast Train", "speed": 80, "hour": 14, "delayed": False, "expected": "Anomaly"},
        {"name": "Very Slow Train", "speed": 5, "hour": 14, "delayed": True, "expected": "Anomaly"},
        {"name": "Late Night Normal", "speed": 35, "hour": 2, "delayed": False, "expected": "Normal"},
        {"name": "Rush Hour Delayed", "speed": 15, "hour": 17, "delayed": True, "expected": "Anomaly"},
    ]
    
    st.markdown("**Sample Predictions for Different Scenarios:**")
    
    results = []
    for scenario in scenarios:
        train_data = {
            "speed_kmh": scenario["speed"],
            "hour_of_day": scenario["hour"],
            "day_of_week": 2,  # Wednesday
            "is_delayed": int(scenario["delayed"]),
            "heading": 180,
            "latitude": 41.8781,
            "longitude": -87.6298,
            "is_weekend": False,
            "is_rush_hour": scenario["hour"] in [7, 8, 9, 17, 18, 19],
            "route_name": "red"
        }
        
        with st.spinner(f"Testing {scenario['name']}..."):
            result = predict_anomaly(train_data)
        
        if result:
            results.append({
                "Scenario": scenario["name"],
                "Speed (km/h)": scenario["speed"],
                "Hour": scenario["hour"],
                "Delayed": scenario["delayed"],
                "Predicted": "Anomaly" if result["is_anomaly"] else "Normal",
                "Probability": f"{result['anomaly_probability']:.1%}",
                "Expected": scenario["expected"]
            })
    
    if results:
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)

def show_live_detection():
    """Live anomaly detection interface"""
    st.header("Live Anomaly Detection")
    st.markdown("Input train parameters and get real-time anomaly predictions")
    
    # Two-column layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🚂 Train Parameters")
        
        # Route selection
        routes = get_supported_routes()
        route_name = st.selectbox("CTA Route", routes, index=routes.index('red') if 'red' in routes else 0)
        
        # Speed input
        speed_kmh = st.number_input("Speed (km/h)", min_value=0.0, max_value=200.0, value=30.0, step=0.1)
        
        # Location inputs
        st.markdown("**Location**")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input("Latitude", min_value=41.6, max_value=42.1, value=41.8781, step=0.0001, format="%.4f")
        with col_lon:
            longitude = st.number_input("Longitude", min_value=-87.9, max_value=-87.5, value=-87.6298, step=0.0001, format="%.4f")
        
        # Other parameters
        heading = st.slider("Heading (degrees)", 0, 360, 180, 1)
        is_delayed = st.checkbox("Train is delayed")
        
        # Time parameters
        st.markdown("**Time Context**")
        hour_of_day = st.slider("Hour of Day", 0, 23, 14, 1)
        day_of_week = st.selectbox("Day of Week", {
            0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
            4: "Friday", 5: "Saturday", 6: "Sunday"
        })
        
        # Auto-calculate derived features
        is_weekend = day_of_week >= 5
        is_rush_hour = hour_of_day in [7, 8, 9, 17, 18, 19]
        
        st.info(f"Weekend: {'Yes' if is_weekend else 'No'} | Rush Hour: {'Yes' if is_rush_hour else 'No'}")
    
    with col2:
        st.subheader("Prediction Results")
        
        # Predict button
        if st.button("🔮 Predict Anomaly", use_container_width=True):
            # Prepare data
            train_data = {
                "speed_kmh": speed_kmh,
                "hour_of_day": hour_of_day,
                "day_of_week": day_of_week,
                "is_delayed": int(is_delayed),
                "heading": heading,
                "latitude": latitude,
                "longitude": longitude,
                "is_weekend": is_weekend,
                "is_rush_hour": is_rush_hour,
                "route_name": route_name
            }
            
            # Make prediction
            with st.spinner("Analyzing train behavior..."):
                result = predict_anomaly(train_data)
            
            if result:
                # Display main result
                if result['is_anomaly']:
                    st.error("**ANOMALY DETECTED**")
                    st.markdown("This train behavior appears **unusual** for the given context.")
                else:
                    st.success("**NORMAL BEHAVIOR**")
                    st.markdown("This train behavior appears **normal** for the given context.")
                
                # Metrics
                col_prob, col_conf = st.columns(2)
                with col_prob:
                    st.metric("Anomaly Probability", f"{result['anomaly_probability']:.1%}")
                with col_conf:
                    st.metric("Model Confidence", f"{result['confidence_score']:.1%}")
                
                # Progress bars for visual appeal
                st.markdown("**Detailed Scores:**")
                st.progress(result['anomaly_probability'], text=f"Anomaly Probability: {result['anomaly_probability']:.1%}")
                st.progress(result['confidence_score'], text=f"Model Confidence: {result['confidence_score']:.1%}")
                
                # Additional info
                with st.expander("Technical Details"):
                    st.markdown(f"**Model Used**: {result['model_used']}")
                    st.markdown(f"**Prediction Time**: {result['timestamp']}")
                    st.json(result['input_features'])
        
        # Quick test buttons
        st.subheader("⚡ Quick Tests")
        
        col_normal, col_anomaly = st.columns(2)
        
        with col_normal:
            if st.button("Test Normal Train", use_container_width=True):
                normal_data = {
                    "speed_kmh": 28.0, "hour_of_day": 14, "day_of_week": 2,
                    "is_delayed": 0, "heading": 180, "latitude": 41.8781,
                    "longitude": -87.6298, "is_weekend": False, "is_rush_hour": False,
                    "route_name": "red"
                }
                result = predict_anomaly(normal_data)
                if result:
                    st.success(f"Normal: {result['anomaly_probability']:.1%} anomaly probability")
        
        with col_anomaly:
            if st.button("Test Anomalous Train", use_container_width=True):
                anomaly_data = {
                    "speed_kmh": 95.0, "hour_of_day": 3, "day_of_week": 2,
                    "is_delayed": 1, "heading": 45, "latitude": 41.8781,
                    "longitude": -87.6298, "is_weekend": False, "is_rush_hour": False,
                    "route_name": "red"
                }
                result = predict_anomaly(anomaly_data)
                if result:
                    st.error(f"Anomaly: {result['anomaly_probability']:.1%} anomaly probability")

if __name__ == "__main__":
    main()