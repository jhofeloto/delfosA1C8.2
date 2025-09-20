#!/usr/bin/env python
"""Train all models"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from pathlib import Path

from src.data.preprocessor import DiabetesDataPreprocessor
from src.models.gradient_boosting import DiabetesGradientBoosting
import json

def train_all_models(X_train, y_train, X_test, y_test):
    models = {
        'linear_regression': LinearRegression(),
        'ridge_regression': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'gradient_boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=7, random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'model': model,
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        print(f"  R²: {results[name]['r2']:.4f}")
    
    return results

def main():
    print("="*60)
    print("TRAINING DIABETES PREDICTION MODELS")
    print("="*60)
    
    preprocessor = DiabetesDataPreprocessor()
    
    data_path = Path("data/raw/diabetes_data.csv")
    if not data_path.exists():
        print("Generating synthetic data...")
        from scripts.generate_synthetic_data import generate_synthetic_data
        df = generate_synthetic_data(100)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
    
    df = preprocessor.load_data(str(data_path))
    df = preprocessor.clean_data(df)
    df = preprocessor.engineer_features(df)
    
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    # Keep feature order used for training (all numeric except target after engineer_features)
    feature_order = list(X_train.columns)

    # Scale features using StandardScaler inside preprocessor
    X_train, X_test = preprocessor.scale_features(X_train, X_test)
    
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Save models and artifacts
    Path("models").mkdir(exist_ok=True)
    metrics = {}
    for name, data in results.items():
        joblib.dump(data['model'], f"models/{name}.pkl")
        print(f"Saved {name}")
        metrics[name] = {
            'r2': data['r2'],
            'rmse': data['rmse'],
            'mae': data['mae']
        }

    # Save metrics metadata
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save scaler and feature order for inference
    joblib.dump(preprocessor.scaler, "models/scaler.pkl")
    with open("models/feature_order.json", "w") as f:
        json.dump({"features": feature_order}, f, indent=2)
    
    best = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nBest Model: {best[0]} (R²={best[1]['r2']:.4f})")

if __name__ == "__main__":
    main()
