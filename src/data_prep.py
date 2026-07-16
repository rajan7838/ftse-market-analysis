import pandas as pd
import numpy as np
import yaml
import joblib
import os
from sklearn.preprocessing import StandardScaler

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_data():
    """Load raw FTSE data"""
    config = load_config()
    df = pd.read_csv(config['data']['raw_path'])
    
    return df

def preprocess_data(df):
    """Clean and preprocess data"""
    print("Starting data preprocessing...")
    
    df = df.ffill()
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    
    df['Return'] = df['Close'].pct_change() * 100
    df['Range'] = ((df['High'] - df['Low']) / df['Low']) * 100
    df = df.dropna()
    
    print(f"Data preprocessing complete: {df.shape[0]} rows")
    return df

def create_features(df):
    """Select and scale features"""
    print("Creating features...")
    config = load_config()
    features = config['features']
    X = df[features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"Features scaled: {X_scaled.shape}")
    
    os.makedirs(os.path.dirname(config['models']['scaler_path']), exist_ok=True)
    joblib.dump(scaler, config['models']['scaler_path'])
    print(f"Scaler saved to {config['models']['scaler_path']}")
    
    return X_scaled, scaler, features

def save_processed_data(df):
    """Save processed data"""
    config = load_config()
    os.makedirs(os.path.dirname(config['data']['processed_path']), exist_ok=True)
    df.to_csv(config['data']['processed_path'])
    print(f"Processed data saved to {config['data']['processed_path']}")

if __name__ == "__main__":
    
    
    df = load_data()
    df = preprocess_data(df)
    X_scaled, scaler, features = create_features(df)
    save_processed_data(df)
    
    print("Data preprocessing complete!")