import os
import warnings
import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
from data_prep import load_data, preprocess_data, create_features

warnings.filterwarnings('ignore')

print("=" * 50)
print("EVALUATION PIPELINE")
print("=" * 50)

# 1. Load Configurations
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 2. Process Raw Historical Data
df = load_data()
df = preprocess_data(df)
X_scaled, scaler_from_prep, features = create_features(df)

# 3. Load Trained Saved Models
print("Loading trained models...")
kmean = joblib.load(config['models']['kmeans_path'])
pca = joblib.load(config['models']['pca_path'])
scaler = joblib.load(config['models']['scaler_path'])
iso_model = joblib.load(config['models']['isolation_forest_path'])
dbscan_labels = np.load('models/dbscan_labels.npy')
print("Models loaded successfully")

# 4. Transform Dataset to PCA space
X_pca = pca.transform(X_scaled)
kmeans_labels = kmean.predict(X_pca)

# 5. Handle Hierarchical Fallback File Check
hier_labels = None
if os.path.exists('models/hierarchical_labels.npy'):
    hier_labels = np.load('models/hierarchical_labels.npy')
    print("Hierarchical labels loaded")
else:
    print("Hierarchical labels not found")

# 6. Predict New Market Dummy Values (Keeping the logic intact)
# Replace these values with your actual test numbers if needed
open_price, high_price, low_price, close_price, volume = 7500.0, 7600.0, 7450.0, 7550.0, 1500000
print(f"Predicting for market: Open={open_price}, Close={close_price}")

new_day = pd.DataFrame({
    'Open': [open_price], 'High': [high_price], 'Low': [low_price], 'Close': [close_price], 'Volume': [volume],
    'Return': [(close_price - open_price) / open_price * 100],
    'Range': ((high_price - low_price) / low_price) * 100
})

new_scaled = scaler.transform(new_day[features])
new_pca = pca.transform(new_scaled)

kmeans_cluster = kmean.predict(new_pca)[0]

distances = distance.cdist(new_pca, X_pca, metric='euclidean')
nearest_idx = np.argmin(distances)
dbscan_cluster = dbscan_labels[nearest_idx]

anomaly_status = iso_model.predict(new_pca)[0]
print(f"Prediction: K-Means={kmeans_cluster}, DBSCAN={dbscan_cluster}, Anomaly={anomaly_status}")

# 7. Model Performance Evaluation Metrics
print("Evaluating models...")
k_score = silhouette_score(X_pca, kmeans_labels)
print(f"K-Means Silhouette Score: {k_score:.3f}")

h_score = 0
if hier_labels is not None:
    h_score = silhouette_score(X_pca, hier_labels)
    print(f"Hierarchical Silhouette Score: {h_score:.3f}")

clean_data = X_pca[dbscan_labels != -1]
clean_labels = dbscan_labels[dbscan_labels != -1]
if len(clean_data) > 0:
    d_score = silhouette_score(clean_data, clean_labels)
    print(f"DBSCAN Silhouette Score: {d_score:.3f}")
else:
    d_score = 0
    print("DBSCAN: No valid clusters found")

# 8. Render Final Output Logs Summary

print("MODEL EVALUATION RESULTS")

print(f"K-Means Score:      {k_score:.3f}")
print(f"Hierarchical Score: {h_score:.3f}")
print(f"DBSCAN Score:       {d_score:.3f}")

