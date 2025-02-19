#!/usr/bin/env python
"""
Script: train.py
----------------
Loads data, initializes the model, trains it, and saves the best model.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from config import MODEL, TRAIN, PATHS
from state_prediction.models import get_model  # Updated import

def main():
    device = torch.device(TRAIN.device)
    print(f"[INFO] Using device: {device}")

    # 1. Load data
    X_train = np.load(PATHS.train_data_dir / 'X_train.npy')
    y_train = np.load(PATHS.train_data_dir / 'y_train.npy')
    X_test  = np.load(PATHS.train_data_dir / 'X_test.npy')
    y_test  = np.load(PATHS.train_data_dir / 'y_test.npy')
    
    # Convert to float32 for PyTorch
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)
    y_test  = y_test.astype(np.float32)

    # 2. Move to torch Tensors
    X_train_t = torch.from_numpy(X_train).float().to(device)
    y_train_t = torch.from_numpy(y_train).float().to(device).squeeze(1)
    X_test_t  = torch.from_numpy(X_test).float().to(device)
    y_test_t  = torch.from_numpy(y_test).float().to(device).squeeze(1)

    # 3. Create DataLoaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=TRAIN.batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=TRAIN.batch_size, shuffle=False)

    # 4. Initialize model via the factory function
    model_config = vars(MODEL).copy()
    model_type = model_config.pop("model_type", "transformer")
    model = get_model(model_type, **model_config).to(device)
    print(model)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=TRAIN.learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-6
    )

    # 5. Training loop
    NUM_EPOCHS = TRAIN.num_epochs
    best_test_loss = float('inf')

    train_losses = []
    test_losses  = []
    epochs_list  = []

    plt.figure()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss  = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for Xb, yb in test_loader:
                preds = model(Xb)
                loss  = criterion(preds, yb)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        train_losses.append(avg_train_loss)
        test_losses.append(avg_test_loss)
        epochs_list.append(epoch+1)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            torch.save(model.state_dict(), PATHS.model_dir / TRAIN.model_filename)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

        scheduler.step(avg_test_loss)

    plt.clf()
    plt.plot(epochs_list, train_losses, label="Train Loss")
    plt.plot(epochs_list, test_losses,  label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.yscale("log")
    plt.title("Transformer Training Losses")
    plt.savefig(PATHS.model_dir / "training_loss_plot.png")

    print("[INFO] Training completed. Best test loss:", best_test_loss)
    print("[INFO] 'best_model.pth' saved in 'model/' folder.")

if __name__ == "__main__":
    main()
 