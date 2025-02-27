#!/usr/bin/env python
"""
Script: predict.py
------------------
Loads the best model, runs evaluations and demonstrates
a sample prediction (or multi-step autoregressive sequence generation).
"""

import numpy as np
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import PATHS, DATA, MODEL, TRAIN, INFERENCE
import argparse
import json

# Add parent directory to path for model imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model

def generate_sequence(model, initial_sequence, y_scaler, num_predictions=None, device=None):
    """
    Autoregressive multi-step generation.
    initial_sequence: (1, seq_len, input_dim) scaled
    """
    num_predictions = num_predictions or INFERENCE.num_generated_steps
    device = device or TRAIN.device
    model.eval()
    if not isinstance(initial_sequence, torch.Tensor):
        initial_sequence = torch.tensor(initial_sequence, dtype=torch.float32)
    initial_sequence = initial_sequence.to(device)

    predictions = []
    current_seq  = initial_sequence.clone()

    with torch.no_grad():
        for _ in range(num_predictions):
            next_pred = model(current_seq)  # (1, target_dim)
            predictions.append(next_pred.cpu().numpy())
            current_seq = current_seq.roll(-1, dims=1)
            current_seq[:, -1, :] = next_pred

    predictions = np.array(predictions).squeeze(axis=1)
    predictions_unscaled = y_scaler.inverse_transform(predictions)
    return predictions_unscaled

def plot_trajectories(actual_sequences, predicted_sequences, save_path):
    """
    Creates a spaghetti plot of multiple actual and predicted trajectories
    
    Args:
        actual_sequences: List of actual trajectory sequences (n_sequences, seq_len, features)
        predicted_sequences: List of predicted trajectory sequences (n_sequences, pred_len, features)
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Plot actual trajectories in light gray
    for seq in actual_sequences:
        plt.plot(seq[:, 0], seq[:, 1], 'gray', alpha=0.3, linewidth=1, label='_nolegend_')
    
    # Plot predicted trajectories
    for i, (actual, pred) in enumerate(zip(actual_sequences, predicted_sequences)):
        # Combine the last point of actual with prediction for continuity
        full_pred = np.vstack([actual[-1:], pred])
        plt.plot(full_pred[:, 0], full_pred[:, 1], 'b-', alpha=0.6, linewidth=1.5, label='Predicted' if i == 0 else '_nolegend_')
        plt.plot(actual[-1, 0], actual[-1, 1], 'go', markersize=6, label='Start Point' if i == 0 else '_nolegend_')
        plt.plot(pred[-1, 0], pred[-1, 1], 'ro', markersize=6, label='End Point' if i == 0 else '_nolegend_')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Multiple Flight Trajectory Predictions')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print("[INFO] Saved multiple flight paths plot to:", save_path)

def parse_args():
    parser = argparse.ArgumentParser(description='Run predictions with trained model')
    parser.add_argument('--model', type=str, choices=['transformer', 'lstm', 'ffnn', 'kalman'],
                       default='transformer', help='Model architecture to use')
    return parser.parse_args()

def load_config(model_dir, model_type):
    """Load model configuration from JSON file"""
    config_path = model_dir / f"{model_type}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    # Parse command line arguments
    args = parse_args()
    
    device = torch.device(TRAIN.device)
    print("[INFO] Using device:", device)

    # Load model configuration for specific model type
    config = load_config(PATHS.get_model_config_path(args.model))
    
    # Verify model type matches
    if config["model_type"] != args.model:
        raise ValueError(f"Requested model type '{args.model}' doesn't match saved model type '{config['model_type']}'")
    
    # Update MODEL config with saved values
    MODEL.model_type = config["model_type"]
    MODEL.input_dim = config["input_dim"]
    model_specific_config = config[f"{MODEL.model_type}_config"]
    
    # Update the specific model configuration
    if MODEL.model_type == "transformer":
        MODEL.transformer = TransformerConfig(**model_specific_config)
    elif MODEL.model_type == "lstm":
        MODEL.lstm = LSTMConfig(**model_specific_config)
    elif MODEL.model_type == "ffnn":
        MODEL.ffnn = FFNNConfig(**model_specific_config)

    # 1. Load scalers
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = pickle.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)

    # 2. Load test data
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test = np.load(PATHS.train_data_dir / 'y_test.npy')

    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    y_test_t = torch.from_numpy(y_test.astype(np.float32)).to(device).squeeze(1)

    test_dataset = TensorDataset(X_test_t, y_test_t)
    test_loader  = DataLoader(test_dataset, batch_size=TRAIN.batch_size, shuffle=False)

    # 3. Load model (using the factory)
    # Build model parameters based on the model type
    if MODEL.model_type.lower() == "transformer":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.transformer)
        }
    elif MODEL.model_type.lower() == "lstm":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.lstm)
        }
    elif MODEL.model_type.lower() == "ffnn":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.ffnn)
        }
    else:
        raise ValueError(f"Unknown model type: {MODEL.model_type}")
    
    model = get_model(MODEL.model_type, **model_params).to(device)
    model.load_state_dict(torch.load(
        PATHS.get_model_weights_path(args.model), 
        map_location=device
    ))
    model.eval()

    criterion = nn.MSELoss()

    total_test_loss = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds = model(Xb)
            loss = criterion(preds, yb)
            total_test_loss += loss.item()
    avg_test_loss = total_test_loss / len(test_loader)
    print(f"[INFO] Test Loss: {avg_test_loss:.6f}")

    # Create visualizations directory if it doesn't exist
    vis_dir = PATHS.model_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate predictions for multiple sequences
    num_sequences = 10  # Number of trajectories to plot
    sequence_indices = np.random.choice(len(X_test), num_sequences, replace=False)
    
    actual_sequences = []
    predicted_sequences = []
    
    for idx in sequence_indices:
        # Get the actual sequence (unscaled)
        actual_seq = X_scaler.inverse_transform(X_test[idx])
        actual_sequences.append(actual_seq)
        
        # Generate prediction with 200 steps
        sample_input = X_test_t[idx:idx+1]
        gen_sequence = generate_sequence(
            model,
            sample_input,
            y_scaler,
            num_predictions=300,  # Changed to generate 200 steps
            device=device
        )
        predicted_sequences.append(gen_sequence)

    # Create the spaghetti plot
    plot_trajectories(
        actual_sequences,
        predicted_sequences,
        vis_dir / "multiple_flight_paths.png"
    )

    sample_input = X_test_t[0:1]
    prediction  = model(sample_input)
    pred_np     = prediction.detach().cpu().numpy()
    pred_unscaled = y_scaler.inverse_transform(pred_np)

    print("\nSingle-step prediction example:")
    print("  Last input time-step (unscaled):")
    last_input_unscaled = X_scaler.inverse_transform(X_test_t[0][-1].cpu().numpy().reshape(1, -1))
    print("    ", last_input_unscaled.flatten())
    print("  Predicted next state (unscaled):")
    print("    ", pred_unscaled.flatten())

    gen_sequence = generate_sequence(
        model, 
        sample_input, 
        y_scaler, 
        num_predictions=INFERENCE.num_generated_steps, 
        device=device
    )
    print(f"\n[INFO] Generated next {INFERENCE.num_generated_steps} steps (unscaled):\n", gen_sequence)

    state_columns = ["lon", "lat", "heading", "velocity", "vertrate", "geoaltitude"]
    lon_idx, lat_idx = 0, 1
    
    full_seq = np.vstack([
        X_scaler.inverse_transform(sample_input[0].cpu()),
        gen_sequence
    ])

    plt.figure(figsize=(10,8))
    plt.plot(full_seq[:, lon_idx], full_seq[:, lat_idx], 'bo-', label='Predicted Path', linewidth=2)
    plt.plot(full_seq[0, lon_idx], full_seq[0, lat_idx], 'go', label='Start', markersize=10)
    plt.plot(full_seq[-1, lon_idx], full_seq[-1, lat_idx], 'ro', label='End', markersize=10)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude') 
    plt.title('Predicted Flight Trajectory')
    plt.grid(True)  
    plt.legend()
    
    plt.savefig(vis_dir / "predicted_flight_path.png", dpi=300, bbox_inches='tight')
    print("[INFO] Saved predicted flight path plot to:", vis_dir / "predicted_flight_path.png")

if __name__ == "__main__":
    main()
