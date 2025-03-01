#!/usr/bin/env python
"""
Script: analyze_xgboost_failures.py
-----------------------------------
Analyzes failures in XGBoost predictions to understand what features
contribute most to prediction errors.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import argparse
from pathlib import Path
import shap
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PATHS, DATA
from train_xgboost import engineer_features, load_raw_sequence_data

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze XGBoost prediction failures')
    parser.add_argument('--target', type=str, required=True,
                       choices=['lon', 'lat', 'heading', 'velocity', 'vertrate', 'geoaltitude'],
                       help='Target variable to analyze')
    parser.add_argument('--num_failures', type=int, default=20,
                       help='Number of worst predictions to analyze (default: 20)')
    parser.add_argument('--error_threshold', type=float, default=None,
                       help='Analyze predictions with errors above this threshold')
    return parser.parse_args()

def load_xgboost_model(target_name):
    """Load the trained XGBoost model for a specific target"""
    model_path = PATHS.model_dir / 'xgboost' / f'{target_name}_model.bin'
    if not model_path.exists():
        raise FileNotFoundError(f"Model for {target_name} not found at {model_path}")
    
    model = xgb.Booster()
    model.load_model(str(model_path))
    
    # Load feature list
    feature_path = PATHS.model_dir / 'xgboost' / f'{target_name}_features.json'
    with open(feature_path, 'r') as f:
        feature_list = json.load(f)
    
    return model, feature_list

def analyze_failures(model, X_df, y_true, y_pred, target_name, num_failures=20, error_threshold=None):
    """
    Analyze the worst prediction failures.
    
    Args:
        model: Trained XGBoost model
        X_df: DataFrame with features
        y_true: True target values
        y_pred: Predicted target values
        target_name: Name of the target variable
        num_failures: Number of worst predictions to analyze
        error_threshold: Analyze predictions with errors above this threshold
    """
    # Calculate absolute errors
    errors = np.abs(y_true - y_pred)
    
    # Get indices of failures
    if error_threshold is not None:
        failure_indices = np.where(errors > error_threshold)[0]
        if len(failure_indices) == 0:
            print(f"[WARNING] No predictions found with errors above {error_threshold}")
            failure_indices = np.argsort(errors)[-num_failures:]
        elif len(failure_indices) > num_failures:
            # If too many failures above threshold, take the worst ones
            sorted_indices = failure_indices[np.argsort(errors[failure_indices])[::-1]]
            failure_indices = sorted_indices[:num_failures]
    else:
        # Get the worst num_failures predictions
        failure_indices = np.argsort(errors)[-num_failures:]
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Create a directory for failure analysis
    failure_dir = PATHS.model_dir / 'xgboost' / 'failure_analysis' / target_name
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary of failures
    failure_data = pd.DataFrame({
        'Index': failure_indices,
        'True_Value': y_true[failure_indices],
        'Predicted_Value': y_pred[failure_indices],
        'Absolute_Error': errors[failure_indices]
    })
    failure_data.to_csv(failure_dir / 'failure_summary.csv', index=False)
    
    # Calculate SHAP values for all failures
    shap_values = explainer.shap_values(X_df.iloc[failure_indices])
    
    # Create summary plot for all failures
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_df.iloc[failure_indices], show=False)
    plt.title(f'SHAP Summary for {target_name} Failures')
    plt.tight_layout()
    plt.savefig(failure_dir / 'failures_shap_summary.png')
    plt.close()
    
    # Create feature importance plot for failures
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_df.iloc[failure_indices], plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importance for {target_name} Failures')
    plt.tight_layout()
    plt.savefig(failure_dir / 'failures_shap_importance.png')
    plt.close()
    
    # Analyze each failure individually
    for i, idx in enumerate(failure_indices):
        # Get SHAP values for this instance
        instance_shap = explainer.shap_values(X_df.iloc[idx:idx+1])
        
        # Create force plot for this instance
        plt.figure(figsize=(14, 6))
        shap.force_plot(
            explainer.expected_value, 
            instance_shap[0], 
            X_df.iloc[idx], 
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot for Failure {i+1}\n'
                 f'True: {y_true[idx]:.4f}, Predicted: {y_pred[idx]:.4f}, Error: {errors[idx]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_force_plot.png')
        plt.close()
        
        # Create waterfall plot for this instance
        plt.figure(figsize=(14, 10))
        shap.waterfall_plot(
            shap.Explanation(
                values=instance_shap[0],
                base_values=explainer.expected_value,
                data=X_df.iloc[idx].values,
                feature_names=X_df.columns
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot for Failure {i+1}\n'
                 f'True: {y_true[idx]:.4f}, Predicted: {y_pred[idx]:.4f}, Error: {errors[idx]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_waterfall_plot.png')
        plt.close()
    
    # Aggregate analysis of feature contributions to errors
    feature_contributions = pd.DataFrame(
        np.abs(shap_values), 
        columns=X_df.columns
    )
    
    # Calculate average absolute SHAP value for each feature across failures
    avg_contributions = feature_contributions.mean().sort_values(ascending=False)
    
    # Plot average feature contributions to errors
    plt.figure(figsize=(12, 8))
    avg_contributions.head(20).plot(kind='bar')
    plt.title(f'Average Feature Contributions to Errors for {target_name}')
    plt.ylabel('Average |SHAP Value|')
    plt.tight_layout()
    plt.savefig(failure_dir / 'average_feature_contributions.png')
    plt.close()
    
    # Save feature contributions to CSV
    avg_contributions.to_csv(failure_dir / 'average_feature_contributions.csv')
    
    print(f"[INFO] Analyzed {len(failure_indices)} failures for {target_name}")
    print(f"[INFO] Top 5 features contributing to errors:")
    for feature, value in avg_contributions.head(5).items():
        print(f"  - {feature}: {value:.6f}")

def main():
    # Parse command line arguments
    args = parse_args()
    target_name = args.target
    
    # Load data
    print("[INFO] Loading sequence data...")
    X_train, y_train, X_test, y_test, X_scaler, y_scaler = load_raw_sequence_data()
    
    # Get target index
    target_idx = DATA.target_columns.index(target_name)
    
    # Load model and feature list
    print(f"[INFO] Loading XGBoost model for target: {target_name}")
    model, feature_list = load_xgboost_model(target_name)
    
    # Engineer features for test data
    print("[INFO] Engineering features from test sequences...")
    X_test_df = engineer_features(X_test, DATA.feature_columns)
    
    # Ensure all expected features are present
    for feat in feature_list:
        if feat not in X_test_df.columns:
            X_test_df[feat] = 0
    
    # Keep only the features used during training
    X_test_df = X_test_df[feature_list]
    
    # Make predictions
    print("[INFO] Making predictions...")
    dtest = xgb.DMatrix(X_test_df)
    y_pred = model.predict(dtest)
    
    # Get actual values for the target
    y_test_flat = y_test.squeeze(1)  # Remove seq_len dimension
    y_actual = y_test_flat[:, target_idx]
    
    # Calculate overall metrics
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_actual, y_pred)
    
    print(f"[INFO] Overall metrics for {target_name}:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    
    # Analyze failures
    print(f"[INFO] Analyzing prediction failures...")
    analyze_failures(
        model, 
        X_test_df, 
        y_actual, 
        y_pred, 
        target_name,
        num_failures=args.num_failures,
        error_threshold=args.error_threshold
    )
    
    print("[INFO] Failure analysis completed")

if __name__ == "__main__":
    main() 