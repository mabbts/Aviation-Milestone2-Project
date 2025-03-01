#!/usr/bin/env python
"""
Script: analyze_model_failures.py
---------------------------------
Analyzes failures in model predictions to understand what features
contribute most to prediction errors using SHAP values.

Works with all model types: transformer, lstm, ffnn, and xgboost.
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
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from config import PATHS, DATA, MODEL, TRAIN

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze model prediction failures')
    parser.add_argument('--model', type=str, required=True,
                       choices=['transformer', 'lstm', 'ffnn', 'xgboost'],
                       help='Model type to analyze')
    parser.add_argument('--target', type=str, default='all',
                       choices=['all', 'lon', 'lat', 'heading', 'velocity', 'vertrate', 'geoaltitude'],
                       help='Target variable to analyze (default: all)')
    parser.add_argument('--num_failures', type=int, default=20,
                       help='Number of worst predictions to analyze (default: 20)')
    parser.add_argument('--error_threshold', type=float, default=None,
                       help='Analyze predictions with errors above this threshold')
    parser.add_argument('--background_samples', type=int, default=100,
                       help='Number of background samples for SHAP (default: 100)')
    return parser.parse_args()

def load_data():
    """Load the test data and scalers"""
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = pickle.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
        
    return X_test, y_test, X_scaler, y_scaler

def load_model(model_type, target=None):
    """Load the trained model"""
    if model_type == 'xgboost':
        if target == 'all':
            # Load all XGBoost models
            models = {}
            xgboost_dir = PATHS.model_dir / 'xgboost'
            for target_name in DATA.target_columns:
                model_path = xgboost_dir / f'{target_name}_model.bin'
                if model_path.exists():
                    model = xgb.Booster()
                    model.load_model(str(model_path))
                    models[target_name] = model
            
            # Load feature list
            feature_path = xgboost_dir / f'{DATA.target_columns[0]}_features.json'
            with open(feature_path, 'r') as f:
                feature_list = json.load(f)
                
            return models, feature_list
        else:
            # Load specific XGBoost model
            model_path = PATHS.model_dir / 'xgboost' / f'{target}_model.bin'
            if not model_path.exists():
                raise FileNotFoundError(f"Model for {target} not found at {model_path}")
            
            model = xgb.Booster()
            model.load_model(str(model_path))
            
            # Load feature list
            feature_path = PATHS.model_dir / 'xgboost' / f'{target}_features.json'
            with open(feature_path, 'r') as f:
                feature_list = json.load(f)
                
            return model, feature_list
    else:
        # Load neural network model
        # First load the model configuration
        config_path = PATHS.get_model_config_path(model_type)
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Update MODEL configuration with loaded values
        MODEL.model_type = model_type
        model_config = config[f"{model_type}_config"]
        
        # Set model parameters
        model_params = {
            "input_dim": MODEL.input_dim,
            **model_config
        }
        
        # Initialize model
        device = torch.device(TRAIN.device)
        model = get_model(model_type, **model_params).to(device)
        
        # Load weights
        model.load_state_dict(torch.load(
            PATHS.get_model_weights_path(model_type),
            map_location=device
        ))
        
        model.eval()  # Set to evaluation mode
        return model, device

def get_predictions(model, X_test, model_type, device=None, target_idx=None):
    """Get model predictions"""
    if model_type == 'xgboost':
        if isinstance(model, dict):
            # Multiple XGBoost models
            predictions = np.zeros((X_test.shape[0], len(DATA.target_columns)))
            for i, target_name in enumerate(DATA.target_columns):
                if target_name in model:
                    # Engineer features for this model
                    from train_xgboost import engineer_features
                    X_test_df = engineer_features(X_test, DATA.feature_columns)
                    dtest = xgb.DMatrix(X_test_df)
                    predictions[:, i] = model[target_name].predict(dtest)
            return predictions
        else:
            # Single XGBoost model
            from train_xgboost import engineer_features
            X_test_df = engineer_features(X_test, DATA.feature_columns)
            dtest = xgb.DMatrix(X_test_df)
            predictions = model.predict(dtest)
            return predictions
    else:
        # Neural network model
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            predictions = model(X_tensor).cpu().numpy()
            
            if target_idx is not None:
                predictions = predictions[:, target_idx]
                
            return predictions

def analyze_failures_nn(model, X_test, y_test, model_type, target_name, target_idx, 
                      num_failures=20, error_threshold=None, background_samples=100):
    """
    Analyze failures for neural network models using SHAP.
    
    Args:
        model: Trained neural network model
        X_test: Test input data
        y_test: Test target data
        model_type: Type of model ('transformer', 'lstm', 'ffnn')
        target_name: Name of the target variable
        target_idx: Index of the target variable
        num_failures: Number of worst predictions to analyze
        error_threshold: Analyze predictions with errors above this threshold
        background_samples: Number of background samples for SHAP
    """
    device = next(model.parameters()).device
    
    # Get predictions
    with torch.no_grad():
        X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(X_tensor).cpu().numpy()
    
    # Get actual values for the target
    y_actual = y_test.squeeze(1)[:, target_idx]
    y_pred = predictions[:, target_idx]
    
    # Calculate errors
    errors = np.abs(y_actual - y_pred)
    
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
    
    # Create directory for failure analysis
    failure_dir = PATHS.model_dir / model_type / 'failure_analysis' / target_name
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary of worst predictions
    worst_predictions = pd.DataFrame({
        'True_Value': y_actual[failure_indices],
        'Predicted_Value': y_pred[failure_indices],
        'Absolute_Error': errors[failure_indices]
    })
    worst_predictions.to_csv(failure_dir / 'worst_predictions.csv', index=False)
    
    # Create SHAP explainer
    # Select a subset of background samples for efficiency
    background_indices = np.random.choice(len(X_test), min(background_samples, len(X_test)), replace=False)
    background = torch.tensor(X_test[background_indices], dtype=torch.float32).to(device)
    
    # Create a wrapper function for the model that returns only the target dimension
    def model_wrapper(x):
        with torch.no_grad():
            return model(x)[:, target_idx:target_idx+1]
    
    # Create SHAP explainer based on model type
    if model_type in ['transformer', 'lstm']:
        explainer = shap.DeepExplainer(model_wrapper, background)
    else:  # FFNN
        explainer = shap.GradientExplainer(model_wrapper, background)
    
    # Calculate SHAP values for failure cases
    failure_samples = torch.tensor(X_test[failure_indices], dtype=torch.float32).to(device)
    shap_values = explainer.shap_values(failure_samples)
    
    # If shap_values is a list (multiple outputs), take the first element
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    
    # Create feature names for visualization
    feature_names = []
    for t in range(X_test.shape[1]):  # For each time step
        for f in DATA.feature_columns:  # For each feature
            feature_names.append(f"{f}_t-{X_test.shape[1]-t}")
    
    # Analyze each failure
    for i, idx in enumerate(range(len(failure_indices))):
        # Create waterfall plot for this instance
        plt.figure(figsize=(14, 10))
        
        # Reshape SHAP values to match feature names
        flat_shap_values = shap_values[idx].reshape(-1)
        flat_features = X_test[failure_indices[idx]].reshape(-1)
        
        # Take top 20 features by absolute SHAP value
        top_indices = np.argsort(np.abs(flat_shap_values))[-20:]
        
        # Create a bar plot of SHAP values
        plt.barh(
            [feature_names[j] for j in top_indices],
            flat_shap_values[top_indices],
            color=['red' if x > 0 else 'blue' for x in flat_shap_values[top_indices]]
        )
        plt.title(f'Top Features for Prediction {i+1}\n'
                 f'True: {y_actual[failure_indices[idx]]:.4f}, '
                 f'Predicted: {y_pred[failure_indices[idx]]:.4f}, '
                 f'Error: {errors[failure_indices[idx]]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_top_features.png')
        plt.close()
    
    # Calculate average absolute SHAP value for each feature
    feature_contributions = pd.DataFrame(
        np.abs(shap_values.reshape(len(failure_indices), -1)),
        columns=feature_names
    )
    
    # Calculate average contribution per feature
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
    
    # Create summary plot for all failures
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values.reshape(len(failure_indices), -1),
        X_test[failure_indices].reshape(len(failure_indices), -1),
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Summary for {target_name} Failures')
    plt.tight_layout()
    plt.savefig(failure_dir / 'shap_summary.png')
    plt.close()
    
    print(f"[INFO] Analyzed {len(failure_indices)} failures for {target_name}")
    print(f"[INFO] Top 5 features contributing to errors:")
    for feature, value in avg_contributions.head(5).items():
        print(f"  - {feature}: {value:.6f}")

def analyze_failures_xgb(model, X_test, y_test, target_name, target_idx, 
                       num_failures=20, error_threshold=None):
    """
    Analyze failures for XGBoost models using SHAP.
    
    Args:
        model: Trained XGBoost model
        X_test: Test input data
        y_test: Test target data
        target_name: Name of the target variable
        target_idx: Index of the target variable
        num_failures: Number of worst predictions to analyze
        error_threshold: Analyze predictions with errors above this threshold
    """
    from train_xgboost import engineer_features
    
    # Engineer features
    X_test_df = engineer_features(X_test, DATA.feature_columns)
    
    # Get predictions
    dtest = xgb.DMatrix(X_test_df)
    y_pred = model.predict(dtest)
    
    # Get actual values for the target
    y_actual = y_test.squeeze(1)[:, target_idx]
    
    # Calculate errors
    errors = np.abs(y_actual - y_pred)
    
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
    
    # Create directory for failure analysis
    failure_dir = PATHS.model_dir / 'xgboost' / 'failure_analysis' / target_name
    failure_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary of worst predictions
    worst_predictions = pd.DataFrame({
        'True_Value': y_actual[failure_indices],
        'Predicted_Value': y_pred[failure_indices],
        'Absolute_Error': errors[failure_indices]
    })
    worst_predictions.to_csv(failure_dir / 'worst_predictions.csv', index=False)
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(model)
    
    # Calculate SHAP values for failure cases
    shap_values = explainer.shap_values(X_test_df.iloc[failure_indices])
    
    # Analyze each failure
    for i, idx in enumerate(range(len(failure_indices))):
        # Create force plot for this instance
        plt.figure(figsize=(14, 6))
        shap.force_plot(
            explainer.expected_value, 
            shap_values[idx], 
            X_test_df.iloc[failure_indices[idx]], 
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Force Plot for Prediction {i+1}\n'
                 f'True: {y_actual[failure_indices[idx]]:.4f}, '
                 f'Predicted: {y_pred[failure_indices[idx]]:.4f}, '
                 f'Error: {errors[failure_indices[idx]]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_force_plot.png')
        plt.close()
        
        # Create waterfall plot for this instance
        plt.figure(figsize=(14, 10))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[idx],
                base_values=explainer.expected_value,
                data=X_test_df.iloc[failure_indices[idx]].values,
                feature_names=X_test_df.columns
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot for Prediction {i+1}\n'
                 f'True: {y_actual[failure_indices[idx]]:.4f}, '
                 f'Predicted: {y_pred[failure_indices[idx]]:.4f}, '
                 f'Error: {errors[failure_indices[idx]]:.4f}')
        plt.tight_layout()
        plt.savefig(failure_dir / f'failure_{i+1}_waterfall_plot.png')
        plt.close()
    
    # Calculate feature contributions
    feature_contributions = pd.DataFrame(
        np.abs(shap_values), 
        columns=X_test_df.columns
    )
    
    # Calculate average absolute SHAP value for each feature
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
    
    # Create summary plot for all failures
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values,
        X_test_df.iloc[failure_indices],
        show=False
    )
    plt.title(f'SHAP Summary for {target_name} Failures')
    plt.tight_layout()
    plt.savefig(failure_dir / 'shap_summary.png')
    plt.close()
    
    print(f"[INFO] Analyzed {len(failure_indices)} failures for {target_name}")
    print(f"[INFO] Top 5 features contributing to errors:")
    for feature, value in avg_contributions.head(5).items():
        print(f"  - {feature}: {value:.6f}")

def main():
    # Parse command line arguments
    args = parse_args()
    model_type = args.model
    target = args.target
    
    # Load data
    print("[INFO] Loading test data...")
    X_test, y_test, X_scaler, y_scaler = load_data()
    
    # Load model
    print(f"[INFO] Loading {model_type} model...")
    model, extra_info = load_model(model_type, target)
    
    # Analyze failures
    if target == 'all':
        # Analyze all targets
        for i, target_name in enumerate(DATA.target_columns):
            print(f"[INFO] Analyzing failures for target: {target_name}")
            
            if model_type == 'xgboost':
                if target_name in model:
                    analyze_failures_xgb(
                        model[target_name], 
                        X_test, 
                        y_test, 
                        target_name, 
                        i,
                        num_failures=args.num_failures,
                        error_threshold=args.error_threshold
                    )
                else:
                    print(f"[WARNING] No model found for target: {target_name}")
            else:
                analyze_failures_nn(
                    model, 
                    X_test, 
                    y_test, 
                    model_type, 
                    target_name, 
                    i,
                    num_failures=args.num_failures,
                    error_threshold=args.error_threshold,
                    background_samples=args.background_samples
                )
    else:
        # Analyze specific target
        target_idx = DATA.target_columns.index(target)
        print(f"[INFO] Analyzing failures for target: {target}")
        
        if model_type == 'xgboost':
            analyze_failures_xgb(
                model, 
                X_test, 
                y_test, 
                target, 
                target_idx,
                num_failures=args.num_failures,
                error_threshold=args.error_threshold
            )
        else:
            analyze_failures_nn(
                model, 
                X_test, 
                y_test, 
                model_type, 
                target, 
                target_idx,
                num_failures=args.num_failures,
                error_threshold=args.error_threshold,
                background_samples=args.background_samples
            )
    
    print("[INFO] Failure analysis completed")

if __name__ == "__main__":
    main() 