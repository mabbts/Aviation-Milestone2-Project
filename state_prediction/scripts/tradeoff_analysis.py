#!/usr/bin/env python
"""
Script: tradeoff_analysis.py
---------------------------
Analyzes trade-offs between different model characteristics and performance metrics.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import json
import time
import sys
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt

from paths import DATA_DIR
from config import DataConfig, PATHS, DATA, MODEL, TRAIN, PathConfig

# Add parent directory to path to find models module
sys.path.append(str(Path(__file__).parent.parent))
from models import get_model

def parse_args():
    """Parse command line arguments"""
    import argparse
    parser = argparse.ArgumentParser(description='Trade-off analysis for state prediction models')
    parser.add_argument('--model', type=str, choices=['transformer', 'lstm', 'ffnn', 'kalman'],
                       default='transformer', help='Model architecture to use')
    parser.add_argument('--tradeoff', type=str, choices=['data_size', 'speed_accuracy', 'model_capacity'],
                       default='data_size', help='Trade-off to analyze')
    return parser.parse_args()

def load_data():
    """Load train and test data"""
    # Load preprocessed data
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    return X_train, y_train, X_test, y_test

def analyze_data_size_tradeoff(model_type, X_train, y_train, X_test, y_test, device):
    """
    Analyze how model performance varies with training data size
    """
    # Define data size fractions to test
    fractions = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    
    # Load model configuration
    config_path = PATHS.get_model_config_path(model_type)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize result lists
    train_times = []
    train_losses = []
    test_losses = []
    rmse_values = []
    mae_values = []
    
    for fraction in fractions:
        print(f"[INFO] Training with {fraction*100:.1f}% of data")
        
        # Determine number of samples to use
        n_samples = int(len(X_train) * fraction)
        
        # Prepare data subset
        subset_indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_train_subset = X_train[subset_indices]
        y_train_subset = y_train[subset_indices]
        
        # Convert to tensors
        X_train_t = torch.from_numpy(X_train_subset).float().to(device)
        y_train_t = torch.from_numpy(y_train_subset).float().to(device).squeeze(1)
        X_test_t = torch.from_numpy(X_test).float().to(device)
        y_test_t = torch.from_numpy(y_test).float().to(device).squeeze(1)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        test_dataset = TensorDataset(X_test_t, y_test_t)
        
        train_loader = DataLoader(train_dataset, batch_size=TRAIN.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=TRAIN.batch_size, shuffle=False)
        
        # Initialize model
        if model_type.lower() == "transformer":
            model_params = {
                "input_dim": MODEL.input_dim,
                **config[f"{model_type}_config"]
            }
        elif model_type.lower() == "lstm":
            model_params = {
                "input_dim": MODEL.input_dim,
                **config[f"{model_type}_config"]
            }
        elif model_type.lower() == "ffnn":
            model_params = {
                "input_dim": MODEL.input_dim,
                **config[f"{model_type}_config"]
            }
        elif model_type.lower() == "kalman":
            model_params = {
                "input_dim": MODEL.input_dim,
                **config[f"{model_type}_config"]
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = get_model(model_type, **model_params).to(device)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.learning_rate)
        
        # Train the model and measure time
        start_time = time.time()
        
        # Simple training loop
        epochs = 15  # Reduced epochs for quicker analysis
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
            
            # Validation phase
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for Xb, yb in test_loader:
                    preds = model(Xb)
                    loss = criterion(preds, yb)
                    val_loss += loss.item() * Xb.size(0)
            
            val_loss /= len(test_dataset)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        train_time = time.time() - start_time
        
        # Evaluate on test set
        model.eval()
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb)
                test_preds.append(preds.cpu().numpy())
                test_targets.append(yb.cpu().numpy())
        
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        
        # Calculate metrics
        mse = mean_squared_error(test_targets, test_preds)
        rmse = sqrt(mse)
        mae = mean_absolute_error(test_targets, test_preds)
        
        # Append results
        train_times.append(train_time)
        train_losses.append(best_val_loss)  # Using best validation loss as proxy for train loss
        test_losses.append(mse)
        rmse_values.append(rmse)
        mae_values.append(mae)
        
        print(f"  Data Fraction: {fraction}, Training Time: {train_time:.2f}s, MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    return {
        "fractions": fractions,
        "train_times": train_times,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "rmse_values": rmse_values,
        "mae_values": mae_values,
        "n_samples": [int(len(X_train) * f) for f in fractions]
    }

def analyze_speed_accuracy_tradeoff(model_type, X_train, y_train, X_test, y_test, device):
    """
    Analyze trade-off between inference speed and prediction accuracy
    by varying model complexity parameters
    """
    # Load model configuration
    config_path = PATHS.get_model_config_path(model_type)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Define the parameter to vary based on model type
    if model_type.lower() == "transformer":
        param_name = "d_model"
        param_values = [32, 64, 128, 256, 512]
    elif model_type.lower() == "lstm":
        param_name = "hidden_dim"
        param_values = [32, 64, 128, 256, 512]
    elif model_type.lower() == "ffnn":
        param_name = "hidden_dim"
        param_values = [32, 64, 128, 256, 512, 1024]
    elif model_type.lower() == "kalman":
        # Kalman filter is parameterized differently
        param_name = "process_noise"
        param_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Convert data to tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device).squeeze(1)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device).squeeze(1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN.batch_size, shuffle=False)
    
    # Initialize result lists
    train_times = []
    inference_times = []
    test_losses = []
    rmse_values = []
    
    for param_value in param_values:
        print(f"[INFO] Testing {param_name} = {param_value}")
        
        # Create model with updated parameter
        model_config = config[f"{model_type}_config"].copy()
        model_config[param_name] = param_value
        
        model_params = {
            "input_dim": MODEL.input_dim,
            **model_config
        }
        
        model = get_model(model_type, **model_params).to(device)
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.learning_rate)
        
        # Train the model and measure time
        start_time = time.time()
        
        # Simple training loop
        epochs = 10  # Reduced epochs for quicker analysis
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
        
        train_time = time.time() - start_time
        
        # Measure inference time
        model.eval()
        inference_start = time.time()
        
        with torch.no_grad():
            for Xb, _ in test_loader:
                _ = model(Xb)
        
        inference_time = time.time() - inference_start
        
        # Evaluate on test set
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb)
                test_preds.append(preds.cpu().numpy())
                test_targets.append(yb.cpu().numpy())
        
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        
        # Calculate metrics
        mse = mean_squared_error(test_targets, test_preds)
        rmse = sqrt(mse)
        
        # Append results
        train_times.append(train_time)
        inference_times.append(inference_time)
        test_losses.append(mse)
        rmse_values.append(rmse)
        
        print(f"  {param_name}: {param_value}, Params: {total_params}, " 
              f"Train Time: {train_time:.2f}s, Inference Time: {inference_time:.4f}s, " 
              f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    return {
        "param_name": param_name,
        "param_values": param_values,
        "train_times": train_times,
        "inference_times": inference_times,
        "test_losses": test_losses,
        "rmse_values": rmse_values
    }

def analyze_model_capacity_tradeoff(model_type, X_train, y_train, X_test, y_test, device):
    """
    Analyze how model capacity (number of parameters) affects 
    training time, inference time, and accuracy
    """
    # Load model configuration
    config_path = PATHS.get_model_config_path(model_type)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Define model structure variations to test
    model_configs = []
    
    if model_type.lower() == "transformer":
        base_config = config[f"{model_type}_config"].copy()
        model_configs = [
            {"d_model": 32, "nhead": 2, "num_encoder_layers": 1},
            {"d_model": 64, "nhead": 4, "num_encoder_layers": 2},
            {"d_model": 128, "nhead": 8, "num_encoder_layers": 3},
            {"d_model": 256, "nhead": 8, "num_encoder_layers": 4},
            {"d_model": 512, "nhead": 8, "num_encoder_layers": 6}
        ]
    elif model_type.lower() == "lstm":
        base_config = config[f"{model_type}_config"].copy()
        model_configs = [
            {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0},
            {"hidden_dim": 64, "num_layers": 1, "dropout": 0.1},
            {"hidden_dim": 128, "num_layers": 2, "dropout": 0.1},
            {"hidden_dim": 256, "num_layers": 2, "dropout": 0.2},
            {"hidden_dim": 512, "num_layers": 3, "dropout": 0.2}
        ]
    elif model_type.lower() == "ffnn":
        base_config = config[f"{model_type}_config"].copy()
        model_configs = [
            {"hidden_dim": 32, "num_layers": 1, "dropout": 0.0},
            {"hidden_dim": 64, "num_layers": 2, "dropout": 0.1},
            {"hidden_dim": 128, "num_layers": 3, "dropout": 0.1},
            {"hidden_dim": 256, "num_layers": 3, "dropout": 0.2},
            {"hidden_dim": 512, "num_layers": 4, "dropout": 0.2}
        ]
    else:
        # For Kalman or other models, return empty results
        return {
            "model_variations": [],
            "param_counts": [],
            "train_times": [],
            "inference_times": [],
            "test_losses": [],
            "rmse_values": []
        }
    
    # Convert data to tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device).squeeze(1)
    X_test_t = torch.from_numpy(X_test).float().to(device)
    y_test_t = torch.from_numpy(y_test).float().to(device).squeeze(1)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN.batch_size, shuffle=False)
    
    # Initialize result lists
    model_variations = []
    param_counts = []
    train_times = []
    inference_times = []
    test_losses = []
    rmse_values = []
    
    for i, model_config in enumerate(model_configs):
        variation_name = f"Variation {i+1}"
        model_variations.append(variation_name)
        print(f"[INFO] Testing {variation_name}: {model_config}")
        
        # Update base config with this variation
        merged_config = base_config.copy()
        merged_config.update(model_config)
        
        # Create model
        model_params = {
            "input_dim": MODEL.input_dim,
            **merged_config
        }
        
        model = get_model(model_type, **model_params).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        param_counts.append(total_params)
        
        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=TRAIN.learning_rate)
        
        # Train the model and measure time
        start_time = time.time()
        
        # Simple training loop
        epochs = 10  # Reduced epochs for quicker analysis
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            for Xb, yb in train_loader:
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
        
        train_time = time.time() - start_time
        train_times.append(train_time)
        
        # Measure inference time
        model.eval()
        inference_start = time.time()
        
        with torch.no_grad():
            for Xb, _ in test_loader:
                _ = model(Xb)
        
        inference_time = time.time() - inference_start
        inference_times.append(inference_time)
        
        # Evaluate on test set
        test_preds = []
        test_targets = []
        
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb)
                test_preds.append(preds.cpu().numpy())
                test_targets.append(yb.cpu().numpy())
        
        test_preds = np.vstack(test_preds)
        test_targets = np.vstack(test_targets)
        
        # Calculate metrics
        mse = mean_squared_error(test_targets, test_preds)
        rmse = sqrt(mse)
        
        test_losses.append(mse)
        rmse_values.append(rmse)
        
        print(f"  {variation_name}: Params: {total_params}, "
              f"Train Time: {train_time:.2f}s, Inference Time: {inference_time:.4f}s, "
              f"MSE: {mse:.6f}, RMSE: {rmse:.6f}")
    
    return {
        "model_variations": model_variations,
        "configs": model_configs,
        "param_counts": param_counts,
        "train_times": train_times,
        "inference_times": inference_times,
        "test_losses": test_losses,
        "rmse_values": rmse_values
    }

def main():
    # Parse arguments
    args = parse_args()
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    # Set device
    device = torch.device(TRAIN.device)
    print(f"[INFO] Using device: {device}")
    
    # Create model directory path based on model type
    model_dir = PATHS.model_dir / args.model
    vis_dir = model_dir / "visualizations" / "tradeoffs"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Run selected trade-off analysis
    if args.tradeoff == 'data_size':
        print(f"[INFO] Analyzing data size vs. accuracy trade-off for {args.model} model")
        results = analyze_data_size_tradeoff(args.model, X_train, y_train, X_test, y_test, device)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Test Error vs. Training Data Size
        ax1.plot(results["n_samples"], results["rmse_values"], 'o-', color='blue', label='RMSE')
        ax1.set_xlabel('Training Set Size (samples)')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Prediction Error vs. Training Data Size')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Time vs. Data Size
        ax2.plot(results["n_samples"], results["train_times"], 'o-', color='red')
        ax2.set_xlabel('Training Set Size (samples)')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time vs. Data Size')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"{args.model}_data_size_tradeoff.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved data size trade-off plot to: {vis_dir / f'{args.model}_data_size_tradeoff.png'}")
        
        # Save results to JSON
        with open(vis_dir / f"{args.model}_data_size_tradeoff.json", 'w') as f:
            json.dump({
                "model": args.model,
                "tradeoff": args.tradeoff,
                "fractions": results["fractions"],
                "n_samples": results["n_samples"],
                "train_times": results["train_times"],
                "test_losses": results["test_losses"],
                "rmse_values": results["rmse_values"],
                "mae_values": results["mae_values"]
            }, f, indent=4)
    
    elif args.tradeoff == 'speed_accuracy':
        print(f"[INFO] Analyzing speed vs. accuracy trade-off for {args.model} model")
        results = analyze_speed_accuracy_tradeoff(args.model, X_train, y_train, X_test, y_test, device)
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Parameter value vs Error
        ax1.plot(results["param_values"], results["rmse_values"], 'o-', color='blue', label='RMSE')
        ax1.set_xlabel(results["param_name"])
        ax1.set_ylabel('RMSE')
        ax1.set_title(f'Error vs. {results["param_name"]}')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Inference Time vs Error
        ax2.scatter(results["inference_times"], results["rmse_values"], color='red', s=80)
        for i, param_value in enumerate(results["param_values"]):
            ax2.annotate(f"{param_value}", 
                        (results["inference_times"][i], results["rmse_values"][i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        ax2.set_xlabel('Inference Time (seconds)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Speed-Accuracy Trade-off')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"{args.model}_speed_accuracy_tradeoff.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved speed-accuracy trade-off plot to: {vis_dir / f'{args.model}_speed_accuracy_tradeoff.png'}")
        
        # Save results to JSON
        with open(vis_dir / f"{args.model}_speed_accuracy_tradeoff.json", 'w') as f:
            json.dump({
                "model": args.model,
                "tradeoff": args.tradeoff,
                "param_name": results["param_name"],
                "param_values": results["param_values"],
                "train_times": results["train_times"],
                "inference_times": results["inference_times"],
                "test_losses": results["test_losses"],
                "rmse_values": results["rmse_values"]
            }, f, indent=4)
    
    elif args.tradeoff == 'model_capacity':
        print(f"[INFO] Analyzing model capacity trade-offs for {args.model} model")
        results = analyze_model_capacity_tradeoff(args.model, X_train, y_train, X_test, y_test, device)
        
        if not results["model_variations"]:
            print(f"[WARNING] Model capacity analysis not implemented for {args.model}")
            return
        
        # Plot results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Model Capacity vs. Error
        ax1.plot(results["param_counts"], results["rmse_values"], 'o-', color='blue')
        for i, variation in enumerate(results["model_variations"]):
            ax1.annotate(variation, 
                        (results["param_counts"][i], results["rmse_values"][i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        ax1.set_xlabel('Number of Parameters')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Model Capacity vs. Error')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training Time vs. Inference Time colored by error
        scatter = ax2.scatter(results["train_times"], results["inference_times"], 
                             c=results["rmse_values"], cmap='viridis', 
                             s=100, alpha=0.7)
        for i, variation in enumerate(results["model_variations"]):
            ax2.annotate(variation, 
                        (results["train_times"][i], results["inference_times"][i]),
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center')
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('Inference Time (seconds)')
        ax2.set_title('Training Time vs. Inference Time')
        ax2.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('RMSE')
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"{args.model}_model_capacity_tradeoff.png", dpi=300, bbox_inches='tight')
        print(f"[INFO] Saved model capacity trade-off plot to: {vis_dir / f'{args.model}_model_capacity_tradeoff.png'}")
        
        # Save results to JSON
        with open(vis_dir / f"{args.model}_model_capacity_tradeoff.json", 'w') as f:
            json.dump({
                "model": args.model,
                "tradeoff": args.tradeoff,
                "model_variations": results["model_variations"],
                "configs": results["configs"],
                "param_counts": results["param_counts"],
                "train_times": results["train_times"],
                "inference_times": results["inference_times"],
                "test_losses": results["test_losses"],
                "rmse_values": results["rmse_values"]
            }, f, indent=4)
    
    print(f"[INFO] Trade-off analysis for {args.model} model completed.")

if __name__ == "__main__":
    main() 