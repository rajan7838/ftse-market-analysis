import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
import os
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

from data_prep import load_data, preprocess_data, create_features

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def perform_pca(X_scaled, config):
    """Perform PCA dimensionality reduction"""
    
    
    pca = PCA()
    X_pca_full = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    
    print(f"PC1 explained: {explained[0]*100:.1f}% of data")
    print(f"PC2 explained: {explained[1]*100:.1f}% of data")
    print(f"PC3 explained: {explained[2]*100:.1f}% of data")
    print(f"First 2 components capture {cumulative[1]*100:.1f}% of variance")
    
    n_95 = np.argmax(cumulative >= 0.95) + 1
    print(f"Need {n_95} components for 95% variance")
    
    pca_2d = PCA(n_components=config['training']['pca_components'])
    X_pca = pca_2d.fit_transform(X_scaled)
    
    os.makedirs(os.path.dirname(config['models']['pca_path']), exist_ok=True)
    joblib.dump(pca_2d, config['models']['pca_path'])
    print(f"PCA model saved to {config['models']['pca_path']}")
    
    return X_pca, pca_2d, explained, cumulative

def train_kmeans(X_pca, config):
    """Train K-Means clustering"""
    
    
    scores = []
    inertia = []
    k_range = range(config['training']['kmeans_range'][0], 
                    config['training']['kmeans_range'][1])
    
    for k in k_range:
        kmean_obj = KMeans(n_clusters=k, random_state=config['training']['random_state'])
        labels = kmean_obj.fit_predict(X_pca)
        inertia.append(kmean_obj.inertia_)
        score = silhouette_score(X_pca, labels)
        scores.append(score)
        print(f"K={k}: Score={score:.3f}")
    
    best_k = k_range[scores.index(max(scores))]
    print(f"Best K: {best_k} (Score: {max(scores):.3f})")
    
    kmean = KMeans(n_clusters=best_k, random_state=config['training']['random_state'])
    cluster_labels = kmean.fit_predict(X_pca)
    
    os.makedirs(os.path.dirname(config['models']['kmeans_path']), exist_ok=True)
    joblib.dump(kmean, config['models']['kmeans_path'])
    print(f"K-Means model saved to {config['models']['kmeans_path']}")
    
    return kmean, cluster_labels, best_k, scores, inertia

def train_hierarchical(X_pca, config):
    """Train Hierarchical clustering"""
    
    
    best_score = -1
    best_link, best_k = 'ward', 3
    
    for link in ['ward', 'complete']:
        for k in [2, 3, 4]:  # ← FIXED: Added the range back
            labels = AgglomerativeClustering(n_clusters=k, linkage=link).fit_predict(X_pca)
            score = silhouette_score(X_pca, labels)
            print(f"{link}: K={k}: Score={score:.3f}")
            
            if score > best_score:
                best_score, best_link, best_k = score, link, k
    
    print(f"Best: {best_link}, K={best_k}, Score={best_score:.3f}")
    
    hier = AgglomerativeClustering(n_clusters=best_k, linkage=best_link).fit_predict(X_pca)
    
    os.makedirs('models', exist_ok=True)
    np.save('models/hierarchical_labels.npy', hier)
    print(f"Hierarchical labels saved to models/hierarchical_labels.npy")
    
    return hier, best_score, best_link, best_k

def train_dbscan(X_pca, config):
    """Train DBSCAN clustering"""
    
    
    best_score = -1
    best_eps, best_min = 1.0, 5
    
    for eps in config['training']['dbscan_eps']:
        for min_samp in config['training']['dbscan_min_samples']:
            labels = DBSCAN(eps=eps, min_samples=min_samp).fit_predict(X_pca)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            if n_clusters >= 2:
                score = silhouette_score(X_pca, labels)
                print(f"eps={eps}, min={min_samp}: Clusters={n_clusters}, Noise={n_noise}, Score={score:.3f}")
                
                if score > best_score:
                    best_score, best_eps, best_min = score, eps, min_samp
    
    print(f"Best: eps={best_eps}, min_samples={best_min}, Score={best_score:.3f}")
    
    dbscan = DBSCAN(eps=best_eps, min_samples=best_min).fit_predict(X_pca)
    n_clusters = len(set(dbscan)) - (1 if -1 in dbscan else 0)
    n_noise = list(dbscan).count(-1)
    
    os.makedirs('models', exist_ok=True)
    np.save('models/dbscan_labels.npy', dbscan)
    print(f"DBSCAN labels saved to models/dbscan_labels.npy")
    
    return dbscan, best_score, best_eps, best_min, n_clusters, n_noise

def train_isolation_forest(X_pca, config):
    """Train Isolation Forest for anomaly detection"""
    
    
    iso_model = IsolationForest(
        contamination=config['training']['contamination'],
        random_state=config['training']['random_state']
    )
    iso_model.fit(X_pca)
    
    anomaly_predictions = iso_model.predict(X_pca)
    n_anomaly = sum(anomaly_predictions == -1)
    
    print(f"Found {n_anomaly} unusual days ({n_anomaly/len(X_pca)*100:.3f}%)")
    
    os.makedirs(os.path.dirname(config['models']['isolation_forest_path']), exist_ok=True)
    joblib.dump(iso_model, config['models']['isolation_forest_path'])
    print(f"Isolation Forest model saved to {config['models']['isolation_forest_path']}")
    
    return iso_model, anomaly_predictions, n_anomaly

def save_visualizations(X_pca, cluster_labels, hier, dbscan, anomaly_predictions, 
                        explained, k_scores, inertia, best_k, best_score, 
                        d_score, n_cluster, n_noise, n_anomaly, winner):
    """Save all visualization plots"""
    
    os.makedirs('static/images', exist_ok=True)
    
    # PCA Visualization using plt.subplot (your updated version)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(['PC1', 'PC2', 'PC3'], explained[:3], color='skyblue')
    plt.title('Information per Component')
    plt.ylabel('Variance Explained')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], color='black', alpha=0.6, s=10)
    plt.title('Data in 2D (PC1 vs PC2)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('static/images/pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Elbow Plot
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 7), inertia, marker='o', color='red', linestyle='--')
    plt.title('Elbow Plot')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/elbow_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # K-Means Clusters
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='rainbow', s=15)
    plt.title(f'K-Means: {best_k} Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Hierarchical Clusters
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hier, cmap='rainbow', s=15)
    plt.title(f'Hierarchical: {best_k} Clusters')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/hierarchical_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # DBSCAN Clusters
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan, cmap='coolwarm', s=15)
    plt.title(f'DBSCAN: {n_cluster} Clusters, {n_noise} Outliers')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/dbscan_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Anomaly Detection
    plt.figure(figsize=(8, 5))
    colors = ['red' if i == -1 else 'blue' for i in anomaly_predictions]
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=15, alpha=0.6)
    plt.title(f'Anomaly Detection: {n_anomaly} Unusual Days')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    plt.savefig('static/images/anomaly_detection.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model Comparison
    plt.figure(figsize=(8, 5))
    bars = plt.bar(['K-Means', 'Hierarchical', 'DBSCAN'], 
                   [k_scores[-1], best_score, d_score], 
                   color=['skyblue', 'lightgreen', 'orange'])
    plt.title(f'Winner: {winner} (Score: {max([k_scores[-1], best_score, d_score]):.3f})')
    plt.ylabel('Silhouette Score (Higher is Better)')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    for bar, score in zip(bars, [k_scores[-1], best_score, d_score]):
        plt.text(bar.get_x() + bar.get_width()/2, score + 0.02, 
                f'{score:.3f}', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("All visualizations saved to static/images/")

def main():
    """Main training pipeline"""
    config = load_config()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('static/images', exist_ok=True)
    
    
    df = load_data()
    df = preprocess_data(df)
    X_scaled, scaler, features = create_features(df)
    
    X_pca, pca_model, explained, cumulative = perform_pca(X_scaled, config)
    
    kmean, cluster_labels, best_k, k_scores, inertia = train_kmeans(X_pca, config)
    
    hier, h_score, best_link, best_k_hier = train_hierarchical(X_pca, config)
    
    dbscan, d_score, best_eps, best_min, n_cluster, n_noise = train_dbscan(X_pca, config)
    
    iso_model, anomaly_predictions, n_anomaly = train_isolation_forest(X_pca, config)
    
    
    
    scores = [k_scores[-1], h_score, d_score]
    names = ['K-Means', 'Hierarchical', 'DBSCAN']
    winner = names[scores.index(max(scores))]
    
    print(f"K-Means Score      : {k_scores[-1]:.3f} ({best_k} Groups)")
    print(f"Hierarchical Score : {h_score:.3f} ({best_k_hier} Groups)")
    print(f"DBSCAN Score       : {d_score:.3f} ({n_cluster} Groups, {n_noise} Outliers)")
    print(f"Anomalies Found    : {n_anomaly} Unusual Days")
    print(f"\nWINNER ALGORITHM: {winner} (Score: {max(scores):.3f})")
    print("="*50)
    
    save_visualizations(X_pca, cluster_labels, hier, dbscan, anomaly_predictions,
                      explained, k_scores, inertia, best_k, h_score,
                      d_score, n_cluster, n_noise, n_anomaly, winner)
    
    
    print("TRAINING COMPLETE!")
    
    print(f"Models saved to: models/")
    print(f"Visualizations saved to: static/images/")
    

if __name__ == "__main__":
    main()
