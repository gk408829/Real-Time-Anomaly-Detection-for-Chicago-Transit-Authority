# **PROJECT PLAN: Enhancing Urban Mobility: A Real-Time Anomaly Detection System for the Chicago Train Network**

# 1 Introduction

## 1.1 Objective

To design, build, and deploy an end-to-end MLOps system that monitors live Chicago Transit Authority (CTA) train data, detects anomalies in real-time using machine learning with rigorous confidence bounds, and visualizes the system's status on a public-facing dashboard.

## 1.2 Impact

This project is impactful because it addresses a critical challenge in urban mobility: ensuring public transportation is reliable and efficient. For millions of commuters, unexpected delays and disruptions can significantly impact their day. While transit agencies report major outages, they often lack the tools to detect subtle, cascading, or emerging issues in real-time.

By building a real-time anomaly detection system, this project delivers tangible impacts:

1. **For Passengers**: It can power next-generation transit apps that provide smarter alerts. Instead of just saying a train is "delayed," it could warn users about an "unusual slowdown" on their line before the official delay is announced, allowing them to reroute and save time.

2. **For Transit Operators**: A live dashboard highlighting anomalous behavior acts as an early warning system. Operators can see a slowdown developing on a map and proactively investigate the cause—be it a signal issue, a mechanical problem, or overcrowding—potentially preventing a minor issue from becoming a major system-wide failure.

3. **For City Planners**: The historical data on anomalies provides a rich dataset for identifying systemic bottlenecks. Planners can use this information to optimize schedules, allocate resources more effectively, and make informed decisions about infrastructure upgrades.

In essence, this project moves beyond simple tracking to provide actionable intelligence, making the transit system more resilient and trustworthy.

## 1.3 Anomaly Detection

Anomaly detection is a machine learning technique used to identify rare items, events, or observations that deviate significantly from the majority of the data. These deviations are often called anomalies, outliers, exceptions, or contaminants.

At its core, the process involves establishing a clear definition of "normal" behavior based on historical data. The system then monitors new, incoming data and flags any points that do not conform to this established norm.

For example, in our transit system:

* **Normal Behavior**: Trains slow down as they approach a station, travel at a consistent speed between stations during off-peak hours, and experience slightly longer travel times during rush hour.

* **Anomalous Behavior**: A train stopping for an extended period between stations, moving significantly slower than usual for that time of day, or deviating from its designated route.

Anomaly detection is a critical tool in many industries, including cybersecurity (detecting network intrusions), finance (flagging fraudulent transactions), and manufacturing (predicting equipment failures).

# 2 Related Works

The task of detecting anomalies in transportation systems has been an active area of research, evolving from statistical methods to complex deep learning architectures.

Early research often focused on statistical and classical time-series models. For instance, methods like ARIMA (Autoregressive Integrated Moving Average) were used to forecast passenger flow or travel times, with significant deviations from the forecast being flagged as anomalies (Williams & Hoel, 2003). While effective for capturing regular, seasonal patterns, these models often struggle with the complex, non-linear dynamics of a real-world transit system.

With the rise of machine learning, tree-based models like Isolation Forest and clustering algorithms like DBSCAN became popular. These methods are effective at identifying outliers in large, multi-dimensional datasets without needing to model the temporal sequence explicitly. For example, research has shown success in using these techniques to detect anomalous bus trajectories or unusual passenger ticketing patterns (Liu, Ting, & Zhou, 2008). However, they can sometimes miss context-dependent anomalies that are only apparent when viewed as part of a sequence.

More recently, the field has been dominated by deep learning, particularly Recurrent Neural Networks (RNNs) like LSTMs and GRUs. These models excel at learning patterns from sequential data, making them a natural fit for vehicle GPS traces. A systematic literature review on anomaly detection in connected vehicles highlighted that LSTMs, CNNs, and Autoencoders are the most commonly used deep learning techniques (Saleh, et al., 2024). They are often trained to predict a vehicle's future state (e.g., location or speed), and a large error between the prediction and reality signals an anomaly. This approach effectively captures the spatiotemporal nature of the problem.

Current research is pushing towards even more sophisticated models and addressing key limitations. This includes using Graph Neural Networks (GNNs) to model the entire transit network as a graph and exploring Transformer-based architectures to capture long-range dependencies. Furthermore, there is a growing emphasis on explainability and real-time performance, ensuring that detection systems are not only accurate but also interpretable and fast enough for real-world operational use.

# 3 Project Timeline

This timeline provides an ambitious but achievable weekly breakdown for completing the project.

* **Week 1: Foundation & Data Collection**

    - Focus: Establishing the project's technical foundation and data pipeline.

    - Process: This week is dedicated to environment setup and building the core data ingestion script. The primary process involves obtaining API credentials, defining a robust database schema in SQLite, and developing a resilient Python script that can run continuously to poll the CTA API. Error handling and logging are critical components. The week concludes with launching the script to begin accumulating a rich historical dataset.

    - Outcome: A stable, automated `fetch_data.py` script and a growing SQLite database with at least 50,000 rows of clean, structured train data.

* **Week 2: Analysis & Baseline**

    - Focus: Deep data understanding and establishing a performance benchmark.

    - Process: We transition from data collection to analysis. The main activities involve a comprehensive exploration of the dataset's temporal and spatial characteristics within a Jupyter Notebook. This analysis will directly inform our first anomaly detection model—a context-aware Z-score baseline. This baseline is critical as it provides the first tangible results and a benchmark against which all future, more complex models will be measured.

    - Outcome: A detailed EDA notebook (`01-EDA.ipynb`) with key visualizations (including GeoPandas maps) and a documented Z-score model that successfully identifies initial anomalies.

* **Week 3: Preprocessing & First Advanced Model**

    - Focus: Building a machine learning-ready dataset and training a powerful baseline model.

    - Process: This week is about formalizing the data preparation process. We will build a reusable preprocessing pipeline that handles missing values, encodes categorical variables, and scales numerical features. This pipeline will then be used to prepare data for our first advanced model, LightGBM. We will set up MLflow to rigorously track this experiment, logging parameters and performance metrics.

    - Outcome: A clean, well-documented modeling notebook (`02-Modeling.ipynb`), a trained LightGBM model artifact, and the initial MLflow experiment logs.

* **Week 4: API Development & Containerization**

    - Focus: Transforming the trained model into a portable, production-ready service.

    - Process: The focus shifts from data science to software engineering. We will build a FastAPI application to serve our trained LightGBM model. This involves creating a `/predict` endpoint that encapsulates the entire prediction logic. The entire application, including all dependencies and the model artifact, will then be containerized using Docker. The week ends with thorough local testing of the Docker container.

    - Outcome: A fully functional, containerized FastAPI service that can serve predictions locally.

* **Week 5: Initial Deployment & Dashboard (Vertical Slice)**

    - Focus: Achieving a full end-to-end, publicly accessible version of the system.

    - Process: This is a critical integration week. The Docker image will be pushed to a container registry (Docker Hub) and deployed to Google Cloud Run, making our API live. Concurrently, we will develop a Streamlit dashboard that consumes this live API. The goal is to create a simple but functional user interface that displays live train data on a map and highlights anomalies.

    - Outcome: A live API endpoint and a deployed Streamlit dashboard, representing the first complete, demonstrable version of the project.

* **Week 6: Advanced Modeling & Conformal Prediction**

    - Focus: Improving predictive accuracy and adding statistical rigor.

    - Process: With the core infrastructure in place, we return to modeling. We will train a more complex, sequence-aware model like an LSTM and compare its performance against the LightGBM baseline in MLflow. The superior model will be selected. The second half of the week involves upgrading the FastAPI service to integrate this new model and implement the conformal prediction logic, providing statistically valid p-values for each anomaly.

    - Outcome: An updated API running the best-performing model, now with the added capability of providing confidence scores for its predictions.

* **Week 7: Automation (CI/CD)**

    - Focus: Automating the testing and deployment workflow.

    - Process: This week is dedicated to implementing MLOps best practices. We will write unit tests for our data processing and API logic. Then, we will build a GitHub Actions workflow that automatically runs these tests, builds the Docker image, and deploys it to Google Cloud Run whenever changes are pushed to the main branch. This eliminates manual deployment steps and ensures reliability.

    - Outcome: A CI/CD pipeline that automates the entire deployment process.

* **Week 8: Monitoring & Final Polish**

    - Focus: Ensuring long-term reliability and preparing the project for presentation.

    - Process: The final week involves setting up a monitoring stack. We will instrument the API with Prometheus metrics and use Grafana to build two dashboards: one for service health (latency, errors) and another for model performance (conformal prediction coverage, false alarm rate). The remaining time will be spent on writing comprehensive documentation, including a system architecture diagram in the `README.md`.

    - Outcome: A fully monitored system and a professional, well-documented GitHub repository ready to be shared.

# 4 Technical Specifications & Dependencies

## 4.1 Environment Configuration

* Environment Manager: `Conda`

* Environment Name: `transit_anomaly`

* Python Version: `3.10`

## 4.2 Core Dependencies (`requirements.txt`)

        # --- Core Data Science & EDA ---
        pandas
        numpy
        jupyterlab
        matplotlib
        seaborn
        scikit-learn

        # --- Geospatial Analysis ---
        geopandas
        shapely

        # --- Data Acquisition & Storage ---
        requests
        python-dotenv
        sqlalchemy
        # Note: sqlite3 is built into Python

        # --- ML Modeling & Experiment Tracking ---
        tensorflow
        mlflow
        lightgbm
        statsmodels # For ARIMA baseline
        # transformers # Optional, for advanced modeling

        # --- Advanced ML ---
        # For Conformal Prediction
        mapie

        # --- Backend API Server ---
        fastapi
        uvicorn[standard]

        # --- Frontend Dashboard ---
        streamlit
        folium

        # --- Infrastructure & Deployment ---
        # These are installed separately, not via pip
        # Docker
        # Docker Compose

## 4.3 External Services & APIs

* Data Source: Chicago Transit Authority (CTA) Train Tracker API

* Cloud Deployment: Google Cloud Run (Primary) or AWS App Runner (Alternative)

* Dashboard Hosting: Streamlit Cloud (Free)

* Container Registry: Docker Hub

# 5 Project Phases

## Phase 1: Data Ingestion & Storage

* **Goal**: Establish a reliable, automated pipeline to collect and store live CTA train data.

* **Tasks**:

    1. API Access: Obtain a CTA Train Tracker API key.

    2. Data Collection Script (`fetch_data.py`): Develop a Python script to poll the ttpositions.aspx endpoint every 60 seconds. Implement a robust loop with error handling and logging. 

    3. Database Schema: Create a local SQLite database (`cta_database.db`) with a clearly defined `train_positions` table.

    4. Data Parsing and Storage: Parse the incoming JSON, convert timestamps to Unix format, and write the cleaned records to the SQLite database.

    5. Initial Data Collection: Run the script in several multi-hour sessions to collect an initial dataset of at least 50,000-100,000 rows.

* **Deliverable**: A Python script that continuously logs live train data into a structured, local SQLite database.

## Phase 2: Exploratory Data Analysis (EDA) & Baseline Modeling

* **Goal**: Understand the dataset's characteristics and establish a simple, statistics-based baseline for detecting point anomalies.

* **Tasks**:

    1. Analysis Environment: Create a Jupyter Notebook (`01-EDA.ipynb`).

    2. Data Loading & Analysis: Load data from SQLite. Perform temporal and spatial analysis, visualizing train patterns and locations using GeoPandas.

    3. Feature Engineering: Calculate train speed (`speed_kmh`) using the Haversine distance.

    4. Baseline Model (Z-Score): Implement a context-aware Z-score model on `speed_kmh`, grouped by `hour_of_day`.

* **Deliverable**: A Jupyter Notebook containing a comprehensive analysis of the data and a working Z-score baseline model.

## Phase 3: Data Preprocessing & Advanced Modeling

* **Goal**: Prepare the data for machine learning and train a sophisticated, interpretable model to detect "Gray Swan" collective anomalies.

* **Tasks**:

    1. Preprocessing Pipeline: Handle missing values, one-hot encode categorical features, and scale numerical features.

    2. Advanced Feature Engineering: Engineer features that capture historical delays and create time-series sequences using a sliding window approach.

    3. Model Selection & Training:

        - Target Variable: Define a clear target, such as `speed_kmh` or `arrival_delay_seconds`.

        - Model Candidates: ARIMA, LightGBM, LSTM/GRU (primary candidate for collective anomalies), Transformer.

    4. Experiment Tracking: Use MLflow to log experiments, compare the performance of different models, and save the best-performing model artifact.

    5. Model Interpretation: Use SHAP to analyze the trained models. Generate global feature importance plots to understand the key drivers of predictions and analyze individual predictions to ensure the model's logic is sound.

* **Deliverable**: A trained model file, an MLflow experiment log, and SHAP analysis plots.

## Phase 4: Backend API with Conformal Prediction & Explainability

* **Goal**: Expose the model as a service that provides explainable predictions with rigorous confidence bounds.

* **Tasks**:

    1. API Development: Build a FastAPI application.

    2. Prediction Logic: Implement the logic to generate predictions, SHAP values for individual requests, and conformal prediction scores.

    3. Online Adaptation: Maintain a rolling calibration buffer for the conformal predictor.

    4. API Endpoints: Create a `/predict` endpoint that returns a JSON object: 
    ```
    {
        "is_anomaly": true, 
        "p_value": 0.02, 
        "severity_score": 0.98, 
        "explanation": {
            "feature1": 0.5, 
            "feature2": -0.2
            }
    }.
    ```

    5. Containerization: Write a `Dockerfile` to package the application.

* **Deliverable**: A runnable Docker image that serves explainable, confidence-scored predictions.

## Phase 5: Cloud Deployment & Automation

* Goal: Deploy the service to the cloud and automate the data pipeline.

* Tasks:

    1. Container Registry: Push the Docker image to Docker Hub.

    2. Cloud Deployment Strategy:

        - Primary Option (Google Cloud Run): Deploy the container to Google Cloud Run. This is the recommended approach due to its simplicity, generous free tier, and fully managed, serverless nature. It scales to zero, ensuring minimal cost.

        - Alternative Option (AWS App Runner): As a direct equivalent, AWS App Runner offers similar container-based, serverless deployment with a free tier. This is a strong alternative if you prefer the AWS ecosystem.

    3. (Optional) Workflow Orchestration: Set up a Prefect workflow to automate the entire pipeline.

* **Deliverable**: A live, public API endpoint that can provide real-time anomaly predictions with severity scores.

## Phase 6: Interactive Dashboard

* Goal: Create a user-friendly, visual frontend for the system.

* Tasks:

    1. UI Development: Build a Streamlit web application.

    2. API Integration & Visualization: The app will call the live FastAPI endpoint and use Folium to display a map with train positions color-coded by anomaly severity.

    3. Display Explanations: When a user clicks on an anomalous train, display a simple chart or text showing the top features that contributed to the anomaly flag (e.g., "Flagged due to: High speed, Low historical delay").

    4. Dashboard Hosting: Deploy the application for free on Streamlit Cloud.

* Deliverable: A public URL to an interactive dashboard showing real-time train anomalies and their explanations.

## Phase 7: Phase 7: Advanced MLOps & Diagnostics

* Goal: Add production-grade features to demonstrate a deep understanding of the MLOps lifecycle.

* Tasks:

    1. CI/CD Pipeline: Create GitHub Actions workflows to automate testing and deployment.

    2. Service Monitoring: Set up Prometheus and Grafana to monitor the API's health (latency, error rate).

    3. Model Monitoring: Create a specific Grafana dashboard to track the conformal predictor's Coverage Rate and the system's False Alarm Rate.

    4. Causal Inference Module (Stretch Goal): Develop a post-detection module that attempts to infer the root cause of a flagged anomaly. A simple implementation could involve a graph-based check: after detecting an anomaly on one train, the system queries the status of nearby trains. If multiple trains are anomalous, it suggests a network-wide issue (e.g., weather); if the anomaly is isolated, it suggests a local issue (e.g., mechanical problem).

* Deliverable: A professional, production-style GitHub repository with automated workflows, advanced monitoring, and diagnostic capabilities.




