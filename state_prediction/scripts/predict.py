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
from state_prediction.models import get_model  # Updated import

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

def main():
    device = torch.device(TRAIN.device)
    print("[INFO] Using device:", device)

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
    model_config = vars(MODEL).copy()
    model_type = model_config.pop("model_type", "transformer")
    model = get_model(model_type, **model_config).to(device)
    model.load_state_dict(torch.load(PATHS.model_dir / TRAIN.model_filename, map_location=device))
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
    
    plt.savefig(PATHS.model_dir / "predicted_flight_path.png", dpi=300, bbox_inches='tight')
    print("[INFO] Saved predicted flight path plot to:", PATHS.model_dir / "predicted_flight_path.png")

if __name__ == "__main__":
    main()
