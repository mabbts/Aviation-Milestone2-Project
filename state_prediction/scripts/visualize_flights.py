#!/usr/bin/env python
"""
Script: visualize_flights.py
----------------------------
Creates aesthetically pleasing visualizations of multiple flight trajectory 
predictions for reports and presentations.

This script can be run independently to generate visualizations using a trained model.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
import pickle
import argparse
import json
import sys

# Add parent directory to path for model imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from config import PATHS, MODEL, TRAIN, INFERENCE, TransformerConfig, LSTMConfig, FFNNConfig

def plot_trajectories(actual_sequences, predicted_sequences, save_path, 
                      title="Multiple Flight Trajectory Predictions",
                      figsize=(12, 10), dpi=300):
    """
    Creates a spaghetti plot of multiple actual and predicted trajectories
    
    Args:
        actual_sequences: List of actual trajectory sequences (n_sequences, seq_len, features)
        predicted_sequences: List of predicted trajectory sequences (n_sequences, pred_len, features)
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size tuple (width, height)
        dpi: Resolution for saved image
    """
    plt.figure(figsize=figsize)
    
    # Plot actual trajectories in light gray
    for seq in actual_sequences:
        plt.plot(seq[:, 0], seq[:, 1], 'gray', alpha=0.3, linewidth=1, label='_nolegend_')
    
    # Plot predicted trajectories
    for i, (actual, pred) in enumerate(zip(actual_sequences, predicted_sequences)):
        # Combine the last point of actual with prediction for continuity
        full_pred = np.vstack([actual[-1:], pred])
        plt.plot(full_pred[:, 0], full_pred[:, 1], 'b-', alpha=0.6, linewidth=1.5, 
                label='Predicted' if i == 0 else '_nolegend_')
        plt.plot(actual[-1, 0], actual[-1, 1], 'go', markersize=6, 
                label='Start Point' if i == 0 else '_nolegend_')
        plt.plot(pred[-1, 0], pred[-1, 1], 'ro', markersize=6, 
                label='End Point' if i == 0 else '_nolegend_')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Saved multiple flight paths plot to: {save_path}")

def plot_enhanced_trajectories(actual_sequences, predicted_sequences, save_path,
                              title="Flight Trajectory Predictions",
                              figsize=(15, 12), dpi=400):
    """
    Creates an aesthetically enhanced visualization of flight trajectories
    with improved styling for reports and presentations.
    
    Args:
        actual_sequences: List of actual trajectory sequences
        predicted_sequences: List of predicted trajectory sequences
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size tuple (width, height)
        dpi: Resolution for saved image
    """
    plt.figure(figsize=figsize)
    
    # Set aesthetic style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Background and figure styling
    ax = plt.gca()
    ax.set_facecolor('#f8f9fa')
    plt.grid(color='white', linestyle='-', linewidth=1)
    
    # Plot actual trajectories
    for seq in actual_sequences:
        plt.plot(seq[:, 0], seq[:, 1], color='#cccccc', alpha=0.2, linewidth=1, label='_nolegend_')
    
    # Color palette for predictions
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(predicted_sequences)))
    
    # Plot predicted trajectories with enhanced styling
    for i, (actual, pred) in enumerate(zip(actual_sequences, predicted_sequences)):
        # Combine the last point of actual with prediction for continuity
        full_pred = np.vstack([actual[-1:], pred])
        
        # Plot trajectory with gradient alpha for direction indication
        for j in range(len(full_pred)-1):
            alpha = 0.4 + 0.6 * (j / (len(full_pred)-1))
            plt.plot(full_pred[j:j+2, 0], full_pred[j:j+2, 1], 
                    color=colors[i], alpha=alpha, linewidth=2,
                    label='Predicted' if (i == 0 and j == 0) else '_nolegend_')
        
        # Start and end points
        plt.plot(actual[-1, 0], actual[-1, 1], 'o', color='#2ecc71', 
                markersize=8, markeredgecolor='white', markeredgewidth=1,
                label='Start Point' if i == 0 else '_nolegend_')
        plt.plot(pred[-1, 0], pred[-1, 1], 'o', color='#e74c3c', 
                markersize=8, markeredgecolor='white', markeredgewidth=1,
                label='End Point' if i == 0 else '_nolegend_')

    plt.xlabel('Longitude', fontsize=12)
    plt.ylabel('Latitude', fontsize=12)
    plt.title(title, fontsize=16, pad=20)
    
    # Improve legend
    legend = plt.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)
    legend.get_frame().set_linewidth(0)
    
    # Add subtle border to the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#dddddd')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    print(f"[INFO] Saved enhanced flight paths visualization to: {save_path}")

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
    current_seq = initial_sequence.clone()

    with torch.no_grad():
        for _ in range(num_predictions):
            next_pred = model(current_seq)  # (1, target_dim)
            predictions.append(next_pred.cpu().numpy())
            current_seq = current_seq.roll(-1, dims=1)
            current_seq[:, -1, :] = next_pred

    predictions = np.array(predictions).squeeze(axis=1)
    predictions_unscaled = y_scaler.inverse_transform(predictions)
    return predictions_unscaled

def create_flight_visualizations(model, X_test, X_test_t, X_scaler, y_scaler, 
                                num_sequences=10, prediction_steps=300, device=None):
    """
    Generate and visualize multiple flight path predictions
    
    Args:
        model: Trained prediction model
        X_test: Test data numpy array
        X_test_t: Test data tensor
        X_scaler: Scaler for input data
        y_scaler: Scaler for output data
        num_sequences: Number of flight sequences to visualize
        prediction_steps: Number of steps to predict for each sequence
        device: Computation device
    
    Returns:
        Tuple of (actual_sequences, predicted_sequences)
    """
    # Create visualizations directory if it doesn't exist
    vis_dir = PATHS.model_dir / "visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Generate predictions for multiple sequences
    sequence_indices = np.random.choice(len(X_test), num_sequences, replace=False)
    
    actual_sequences = []
    predicted_sequences = []
    
    for idx in sequence_indices:
        # Get the actual sequence (unscaled)
        actual_seq = X_scaler.inverse_transform(X_test[idx])
        actual_sequences.append(actual_seq)
        
        # Generate prediction
        sample_input = X_test_t[idx:idx+1]
        gen_sequence = generate_sequence(
            model,
            sample_input,
            y_scaler,
            num_predictions=prediction_steps,
            device=device
        )
        predicted_sequences.append(gen_sequence)

    # Create the standard plot
    plot_trajectories(
        actual_sequences,
        predicted_sequences,
        vis_dir / "multiple_flight_paths.png"
    )
    
    # Create the enhanced visualization for reports
    plot_enhanced_trajectories(
        actual_sequences,
        predicted_sequences,
        vis_dir / "enhanced_flight_paths.png",
        title="Flight Trajectory Simulation"
    )
    
    return actual_sequences, predicted_sequences

def parse_args():
    parser = argparse.ArgumentParser(description='Generate flight path visualizations')
    parser.add_argument('--model', type=str, choices=['transformer', 'lstm', 'ffnn', 'kalman'],
                       default='transformer', help='Model architecture to use')
    parser.add_argument('--num_sequences', type=int, default=10,
                       help='Number of flight sequences to visualize')
    parser.add_argument('--prediction_steps', type=int, default=300,
                       help='Number of prediction steps for each sequence')
    return parser.parse_args()

def load_config(model_dir):
    """Load model configuration from JSON file"""
    config_path = model_dir / f"{MODEL.model_type}_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def main():
    args = parse_args()
    device = torch.device(TRAIN.device)
    print("[INFO] Using device:", device)
    
    # Set model type from args
    MODEL.model_type = args.model
    
    # Load model configuration for specific model type
    config_path = PATHS.get_model_config_path(args.model)
    if not config_path.exists():
        raise FileNotFoundError(f"No config file found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Verify model type matches
    if config["model_type"] != args.model:
        raise ValueError(f"Requested model type '{args.model}' doesn't match saved model type '{config['model_type']}'")
    
    # Update MODEL config with saved values
    MODEL.input_dim = config["input_dim"]
    model_specific_config = config[f"{MODEL.model_type}_config"]
    
    # Update the specific model configuration
    if MODEL.model_type == "transformer":
        MODEL.transformer = TransformerConfig(**model_specific_config)
    elif MODEL.model_type == "lstm":
        MODEL.lstm = LSTMConfig(**model_specific_config)
    elif MODEL.model_type == "ffnn":
        MODEL.ffnn = FFNNConfig(**model_specific_config)
    
    # Load scalers
    with open(PATHS.scalers_dir / 'X_scaler.pkl', 'rb') as f:
        X_scaler = pickle.load(f)
    with open(PATHS.scalers_dir / 'y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)
    
    # Load test data
    X_test = np.load(PATHS.train_data_dir / 'X_test.npy')
    X_test_t = torch.from_numpy(X_test.astype(np.float32)).to(device)
    
    # Load model
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
    
    print(f"[INFO] Generating visualizations for {args.num_sequences} flight sequences")
    print(f"[INFO] Each sequence will predict {args.prediction_steps} steps")
    
    # Generate visualizations
    create_flight_visualizations(
        model, 
        X_test, 
        X_test_t, 
        X_scaler, 
        y_scaler,
        num_sequences=args.num_sequences,
        prediction_steps=args.prediction_steps,
        device=device
    )
    
    print("[INFO] Visualization complete!")

if __name__ == "__main__":
    main() 