import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from scipy.spatial import distance

warnings.filterwarnings('ignore')

# Handle dynamic paths for your project scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_prep import create_features, load_data, preprocess_data

# Page configurations
st.set_page_config(page_title="FTSE Market Analysis", layout="wide")
st.title("FTSE Market Analysis")

# Load Configuration file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Cached function to securely load machine learning assets
@st.cache_resource
def load_models():
    kmean = joblib.load('models/kmeans_model.pkl')
    pca = joblib.load('models/pca_model.pkl')
    scaler = joblib.load('models/scaler_model.pkl')
    iso = joblib.load('models/isolation_forest_model.pkl')
    dbscan_labels = np.load('models/dbscan_labels.npy')
    return kmean, pca, scaler, iso, dbscan_labels

# Sidebar configuration for user data input
st.sidebar.header("Input Data")
op = st.sidebar.number_input("Open", value=7500.0)
hi = st.sidebar.number_input("High", value=7600.0)
lo = st.sidebar.number_input("Low", value=7450.0)
cl = st.sidebar.number_input("Close", value=7550.0)
vol = st.sidebar.number_input("Volume", value=1500000)

# Build UI Columns
col1, col2 = st.columns(2)

# Left Column: Rendering Saved Plots
with col1:
    st.markdown("### Visualizations")
    plots = [
        'pca_analysis.png', 
        'kmeans_clusters.png', 
        'dbscan_clusters.png', 
        'anomaly_detection.png', 
        'model_comparison.png'
    ]
    for plot in plots:
        try:
            st.image(f'static/images/{plot}', use_column_width=True)
        except Exception:
            pass

# Right Column: Main Data Pipeline & Metrics Interface
with col2:
    st.markdown("### Market Analysis")
    
    if st.button("Analyze"):
        # Load pre-trained models
        kmean, pca, scaler, iso, dbscan_labels = load_models()
        
        # Load and process the historical dataset
        df = load_data() 
        df = preprocess_data(df)
        X_scaled, _, _ = create_features(df)
        X_pca = pca.transform(X_scaled)
        
        # Format the user inputs into a new single-row DataFrame
        new_day = pd.DataFrame({
            'Open': [op], 
            'High': [hi], 
            'Low': [lo], 
            'Close': [cl], 
            'Volume': [vol], 
            'Return': [(cl - op) / op * 100], 
            'Range': ((hi - lo) / lo) * 100
        })
        
        # Normalize and reduce dimensions for the new data point
        new_scaled = scaler.transform(new_day[config['features']])
        new_pca = pca.transform(new_scaled)
        
        # 1. Compute K-Means prediction
        kmeans_cluster = kmean.predict(new_pca)[0]
        
        # 2. Compute DBSCAN projection using nearest Euclidean distance mapping
        distances = distance.cdist(new_pca, X_pca, metric='euclidean')
        nearest_index = np.argmin(distances)
        dbscan_cluster = dbscan_labels[nearest_index]
        
        # 3. Compute Isolation Forest anomaly status
        anomaly_status = iso.predict(new_pca)[0]
        
        # Display Metrics Dashboard Row
        col_a, col_b, col_c = st.columns(3)
        
        dbscan_text = "Normal" if dbscan_cluster != -1 else "Outlier"
        anomaly_text = "Normal" if anomaly_status != -1 else "Anomaly"
        
        col_a.metric("K-Means", kmeans_cluster)
        col_b.metric("DBSCAN", dbscan_text)
        col_c.metric("Anomaly", anomaly_text)
        
        # Process trading suggestions rules
        if dbscan_cluster == -1 or anomaly_status == -1:
            st.error("Unusual day! Be cautious.")
        elif kmeans_cluster == 1:
            st.success("High activity! Consider aggressive trading.")
        else:
            st.info("Normal day! Follow standard strategy.")
