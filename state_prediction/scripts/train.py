#!/usr/bin/env python
"""
Script: train.py
----------------
Main training script for the state prediction model.

This script handles:
1. Loading and preprocessing training data
2. Setting up model architecture (Transformer or LSTM)
3. Training loop with validation
4. Learning rate scheduling and early stopping
5. Model checkpointing and loss visualization

The script uses configurations from config.py for model architecture, 
training parameters, and file paths.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import pickle
import argparse
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config import MODEL, TRAIN, PATHS

# Add parent directory to path for model imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser(description='Train state prediction model')
    parser.add_argument('--model', type=str, choices=['transformer', 'lstm', 'ffnn', 'kalman'],
                       default='transformer', help='Model architecture to use')
    return parser.parse_args()

def save_config(config_dict, model_dir, model_type):
    """Save model configuration to JSON file"""
    config_path = model_dir / f"{model_type}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    print(f"[INFO] Saved {model_type} configuration to {config_path}")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Update model type in config
    MODEL.model_type = args.model
    
    # Create config dictionary to save
    config_dict = {
        "model_type": MODEL.model_type,
        "input_dim": MODEL.input_dim,
        f"{MODEL.model_type}_config": vars(getattr(MODEL, MODEL.model_type))
    }
    
    # Create model directory if it doesn't exist
    PATHS.model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration with model-specific name
    config_path = PATHS.get_model_config_path(MODEL.model_type)
    save_config(config_dict, PATHS.model_dir, MODEL.model_type)
    
    # Set up device (GPU if available, else CPU)
    device = torch.device(TRAIN.device)
    print(f"[INFO] Using device: {device}")

    # 1. Load and preprocess data
    # Load numpy arrays containing training and test data
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    X_test  = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test  = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    # Convert arrays to float32 for PyTorch compatibility
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    # 2. Convert numpy arrays to PyTorch tensors and move to appropriate device
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device).squeeze(1)
    X_test_t  = torch.from_numpy(X_test).float().to(device)
    y_test_t  = torch.from_numpy(y_test).float().to(device).squeeze(1)

    # 3. Create DataLoader objects for batch processing
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=TRAIN.batch_size, shuffle=False)

    # 4. Initialize model architecture
    # Configure model parameters based on selected architecture
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
    elif MODEL.model_type.lower() == "kalman":
        model_params = {
            "input_dim": MODEL.input_dim,
            **vars(MODEL.kalman)
        }
    else:
        raise ValueError(f"Unknown model type: {MODEL.model_type}")
    
    # Initialize model and move to device
    model = get_model(MODEL.model_type, **model_params).to(device)
    print(model)

    # Set up loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(), 
        lr=TRAIN.learning_rate,
        weight_decay=MODEL.lstm.l2_weight_decay  # L2 regularization
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=TRAIN.lr_factor,
        patience=TRAIN.lr_patience,
        min_lr=TRAIN.min_lr,
        verbose=True
    )

    # Early stopping setup
    best_val_loss = float('inf')
    patience_counter = 0

    # 5. Training loop
    NUM_EPOCHS = TRAIN.num_epochs
    train_losses = []
    test_losses  = []
    epochs_list  = []

    plt.figure()
    
    # Main training loop
    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        total_train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss  = criterion(preds, yb)
            loss.backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb)
                loss  = criterion(preds, yb)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        # Record metrics
        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        epochs_list.append(epoch+1)

        # Learning rate scheduling
        if TRAIN.use_lr_scheduler:
            scheduler.step(avg_test_loss)
        
        # Early stopping check
        if avg_test_loss < best_val_loss - TRAIN.min_delta:
            best_val_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            if TRAIN.save_model:
                torch.save(model.state_dict(), PATHS.get_model_weights_path(MODEL.model_type))
        else:
            patience_counter += 1
            
        if patience_counter >= TRAIN.patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # Plot training history
    plt.clf()
    plt.plot(epochs_list, train_losses, label="Train Loss")
    plt.plot(epochs_list, test_losses,  label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.yscale("log")
    plt.title(f"{MODEL.model_type.capitalize()} Training Losses")
    plt.savefig(PATHS.get_loss_plot_path(MODEL.model_type))

    print("[INFO] Training completed. Best test loss:", best_val_loss)
    print(f"[INFO] Best model saved as '{MODEL.model_filename}' in 'model/' folder.")

if __name__ == "__main__":
    main()