#!/usr/bin/env python
"""
Script: sensitivity_analysis.py
------------------------------
Performs comprehensive sensitivity analysis for LSTM model by testing multiple hyperparameters
and saving results after each parameter test.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import time
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from itertools import product
import sys
from paths import DATA_DIR
from config import DataConfig, PATHS, DATA, MODEL, TRAIN, PathConfig
sys.path.append(str(Path(__file__).parent.parent))
from models import get_model

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='LSTM sensitivity analysis for state prediction')
    parser.add_argument('--params', type=str, default='all',
                       help='Parameters to analyze: "all" or comma-separated list (e.g., "hidden_dim,num_layers")')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of epochs to train each model')
    return parser.parse_args()

def load_data():
    """Load train and test data"""
    # Load preprocessed data
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    return X_train, y_train, X_test, y_test

def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs):
    """
    Train model for specified number of epochs and return training history
    """
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * Xb.size(0)
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb)
                loss = criterion(preds, yb)
                total_test_loss += loss.item() * Xb.size(0)
        
        avg_test_loss = total_test_loss / len(test_loader.dataset)
        test_losses.append(avg_test_loss)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.6f}, Test Loss = {avg_test_loss:.6f}")
    
    return {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1]
    }

def analyze_parameter(param_name, param_values, model_config, X_train_t, y_train_t, X_test_t, y_test_t, 
                     device, epochs, vis_dir):
    """Analyze a single parameter's impact on model performance"""
    print(f"\n[INFO] Running sensitivity analysis for LSTM model on parameter '{param_name}'")
    print(f"[INFO] Testing values: {param_values}")
    
    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    results = []
    
    for value in param_values:
        print(f"[INFO] Testing {param_name} = {value}")
        
        # Create model with updated parameter
        current_config = model_config.copy()
        
        if param_name == "learning_rate":
            # Learning rate is an optimizer parameter
            lr = value
            current_config = model_config.copy()
        elif param_name == "batch_size":
            # Batch size is a loader parameter
            batch_size = value
            current_config = model_config.copy()
        else:
            # For model-specific parameters
            current_config[param_name] = value
        
        # Create model instance
        model_params = {
            "input_dim": MODEL.input_dim,
            **current_config
        }
        model = get_model("lstm", **model_params).to(device)
        
        # Create data loaders
        if param_name == "batch_size":
            train_loader = DataLoader(train_dataset, batch_size=value, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=value, shuffle=False)
        else:
            train_loader = DataLoader(train_dataset, batch_size=TRAIN.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=TRAIN.batch_size, shuffle=False)
        
        # Define loss function
        criterion = nn.MSELoss()
        
        # Define optimizer
        if param_name == "learning_rate":
            optimizer = torch.optim.Adam(model.parameters(), lr=value)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.learning_rate)
        
        # Train model and get history
        start_time = time.time()
        history = train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs)
        training_time = time.time() - start_time
        
        # Save results
        results.append({
            "param_value": value,
            "train_losses": history["train_losses"],
            "test_losses": history["test_losses"],
            "final_train_loss": history["final_train_loss"],
            "final_test_loss": history["final_test_loss"],
            "training_time": training_time
        })
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Convert to strings for plotting
    str_values = [str(v) for v in param_values]
    train_losses = [r["final_train_loss"] for r in results]
    test_losses = [r["final_test_loss"] for r in results]
    
    x = range(len(param_values))
    plt.plot(x, train_losses, 'o-', label='Training Loss')
    plt.plot(x, test_losses, 'o-', label='Test Loss')
    plt.xticks(x, str_values)
    plt.xlabel(param_name)
    plt.ylabel('MSE Loss')
    plt.title(f'LSTM Sensitivity Analysis: {param_name}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(vis_dir / f"lstm_{param_name}_sensitivity.png", dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved sensitivity plot to: {vis_dir / f'lstm_{param_name}_sensitivity.png'}")
    
    # Plot learning curves for each value
    plt.figure(figsize=(14, 10))
    for i, result in enumerate(results):
        plt.subplot(2, len(results)//2 + len(results)%2, i+1)
        plt.plot(result["train_losses"], label='Train')
        plt.plot(result["test_losses"], label='Test')
        plt.title(f'{param_name}={param_values[i]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(vis_dir / f"lstm_{param_name}_learning_curves.png", dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved learning curves to: {vis_dir / f'lstm_{param_name}_learning_curves.png'}")
    
    # Save results to JSON
    sensitivity_results = {
        "model": "lstm",
        "parameter": param_name,
        "values": param_values if not isinstance(param_values[0], (int, float)) else [float(v) for v in param_values],
        "results": [{
            "param_value": float(r["param_value"]) if isinstance(r["param_value"], (int, float)) else r["param_value"],
            "final_train_loss": r["final_train_loss"],
            "final_test_loss": r["final_test_loss"],
            "training_time": r["training_time"]
        } for r in results]
    }
    
    with open(vis_dir / f"lstm_{param_name}_sensitivity.json", 'w') as f:
        json.dump(sensitivity_results, f, indent=4)
    
    print(f"[INFO] Saved sensitivity results to: {vis_dir / f'lstm_{param_name}_sensitivity.json'}")
    
    return results

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Set device
    device = torch.device(TRAIN.device)
    print(f"[INFO] Using device: {device}")
    
    # Create model directory path for LSTM
    model_dir = PATHS.model_dir / "lstm"
    vis_dir = model_dir / "visualizations" / "sensitivity"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load model configuration
    config_path = PATHS.get_model_config_path("lstm")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    lstm_config = config["lstm_config"].copy()
    
    # Define parameter ranges to test
    param_ranges = {
        "hidden_dim": [32, 64, 128, 256, 512],
        "num_layers": [1, 2, 3, 4],
        "dropout": [0.0, 0.1, 0.2, 0.3, 0.5],
        "learning_rate": [1e-2, 1e-3, 1e-4, 1e-5],
        "batch_size": [16, 32, 64, 128, 256]
    }
    
    # Determine which parameters to analyze
    if args.params.lower() == 'all':
        params_to_analyze = list(param_ranges.keys())
    else:
        params_to_analyze = [p.strip() for p in args.params.split(',')]
        # Validate parameters
        for param in params_to_analyze:
            if param not in param_ranges:
                print(f"[WARNING] Unknown parameter: {param}. Skipping.")
                params_to_analyze.remove(param)
    
    print(f"[INFO] Starting comprehensive LSTM sensitivity analysis")
    print(f"[INFO] Parameters to analyze: {params_to_analyze}")
    
    # Convert data to PyTorch tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device).squeeze(1)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device).squeeze(1)
    
    # Store all results
    all_results = {}
    
    # Analyze each parameter
    for param in params_to_analyze:
        print(f"\n{'='*80}")
        print(f"[INFO] Analyzing parameter: {param}")
        print(f"{'='*80}")
        
        param_results = analyze_parameter(
            param, 
            param_ranges[param], 
            lstm_config, 
            X_train_t, y_train_t, 
            X_test_t, y_test_t, 
            device, 
            args.epochs,
            vis_dir
        )
        
        all_results[param] = param_results
    
    # Create summary plot comparing all parameters
    plt.figure(figsize=(15, 10))
    
    # For each parameter, plot the best test loss achieved
    best_values = {}
    for param in params_to_analyze:
        param_results = all_results[param]
        test_losses = [r["final_test_loss"] for r in param_results]
        best_idx = np.argmin(test_losses)
        best_value = param_ranges[param][best_idx]
        best_loss = test_losses[best_idx]
        best_values[param] = (best_value, best_loss)
    
    # Plot best values
    params = list(best_values.keys())
    best_vals = [str(best_values[p][0]) for p in params]
    best_losses = [best_values[p][1] for p in params]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(params, best_losses)
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                f'{best_vals[i]}', ha='center', va='bottom', rotation=0)
    
    plt.ylabel('Best Test Loss (MSE)')
    plt.title('LSTM Sensitivity Analysis: Best Parameter Values')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(vis_dir / "lstm_sensitivity_summary.png", dpi=300, bbox_inches='tight')
    print(f"[INFO] Saved summary plot to: {vis_dir / 'lstm_sensitivity_summary.png'}")
    
    # Save comprehensive results
    with open(vis_dir / "lstm_sensitivity_comprehensive.json", 'w') as f:
        json.dump({
            "model": "lstm",
            "parameters_analyzed": params_to_analyze,
            "best_parameters": {p: float(v[0]) if isinstance(v[0], (int, float)) else v[0] 
                               for p, v in best_values.items()},
            "best_test_losses": {p: v[1] for p, v in best_values.items()}
        }, f, indent=4)
    
    print(f"[INFO] Saved comprehensive results to: {vis_dir / 'lstm_sensitivity_comprehensive.json'}")
    print("[INFO] LSTM sensitivity analysis completed.")

if __name__ == "__main__":
    main() 