import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import ray
from ray.train import ScalingConfig, RunConfig, Checkpoint
from ray.train.torch import TorchTrainer
import ray.train.torch
from tqdm import tqdm

# 1. INITIALIZE RAY INSIDE THE SCRIPT
ray.init(address="auto", namespace="OBIS_TEST", ignore_reinit_error=True)

model_output_path = '/mnt/shared_data/finflow/lstm_models_multi'
train_dir = '/mnt/shared_data/finflow/precomputed_sequences_multi_train'
val_dir = '/mnt/shared_data/finflow/precomputed_sequences_multi_val'
os.makedirs(model_output_path, exist_ok=True)

# ==========================================
# 2. DISTRIBUTED DATA PIPELINE
# ==========================================
print("Loading precomputed multivariate pipelines...")

# No on-the-fly generation needed! Just point and shoot.
train_ds = ray.data.read_parquet(train_dir)
val_ds = ray.data.read_parquet(val_dir)

# ==========================================
# 3. DISTRIBUTED TRAINING WORKER FUNCTION
# ==========================================
def train_loop_per_worker(config):
    class LSTMModel(nn.Module):
        # ---> UPGRADE: input_size=3 <---
        def __init__(self, input_size=3, hidden_size=128, num_layers=2, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                               batch_first=True, dropout=dropout if num_layers > 1 else 0)
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            last_hidden = lstm_out[:, -1, :]
            out = self.fc(last_hidden)
            return out.squeeze()

    # Initialize Multivariate Model
    model = LSTMModel(input_size=3, hidden_size=128, num_layers=2, dropout=0.2)

    model = torch.compile(model)
    model = ray.train.torch.prepare_model(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_shard = ray.train.get_dataset_shard("train")
    val_shard = ray.train.get_dataset_shard("val")

    epochs = config["epochs"]
    batch_size = config["batch_size"]
    world_rank = ray.train.get_context().get_world_rank()

    best_val_loss = float('inf')
    patience, patience_counter = 8, 0
    history = {'epochs': [], 'train_loss': [], 'val_loss': []}

    if world_rank == 0:
        epoch_iterator = tqdm(range(epochs), desc="Training MULTI on 4 GPUs")
    else:
        epoch_iterator = range(epochs)

    for epoch in epoch_iterator:
        # --- Train ---
        model.train()
        train_loss, train_batches = 0, 0

        for batch in train_shard.iter_torch_batches(batch_size=batch_size, local_shuffle_buffer_size=10000):
            # ---> UPGRADE: Snap the flat 36 floats back into (Batch, 12, 3) <---
            X_batch = batch["X"].float().view(-1, 12, 3)
            y_batch = batch["y"].float()

            X_batch = torch.log1p(X_batch) 
            y_batch = torch.log1p(y_batch)

            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = model(X_batch)
                loss = loss_fn(pred, y_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        train_loss /= max(train_batches, 1)

        # --- Validation ---
        model.eval()
        val_loss, val_batches = 0, 0
        with torch.no_grad():
            for batch in val_shard.iter_torch_batches(batch_size=batch_size):
                # ---> UPGRADE: Snap the flat 36 floats back into (Batch, 12, 3) <---
                X_val_batch = batch["X"].float().view(-1, 12, 3)
                y_val_batch = batch["y"].float()

                X_val_batch = torch.log1p(X_val_batch)
                y_val_batch = torch.log1p(y_val_batch)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    val_pred = model(X_val_batch)
                    batch_loss = loss_fn(val_pred, y_val_batch)

                val_loss += batch_loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        if world_rank == 0:
            epoch_iterator.set_postfix({"Train Loss": f"{train_loss:.4f}", "Val Loss": f"{val_loss:.4f}"})
            history['epochs'].append(epoch + 1)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            pd.DataFrame(history).to_csv(f'{config["output_path"]}/training_history_multi.csv', index=False)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.module.state_dict(), f'{config["output_path"]}/best_model_multi.pt')
            else:
                patience_counter += 1

        ray.train.report({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch + 1})

        if patience_counter >= patience:
            break

# ==========================================
# 4. LAUNCH TRAINER (DIRECTLY)
# ==========================================
print("Starting 4-GPU Distributed Trainer...")

scaling_config = ScalingConfig(num_workers=4, use_gpu=True)

trainer = TorchTrainer(
    train_loop_per_worker=train_loop_per_worker,
    train_loop_config={
        "epochs": 150, 
        "batch_size": 1024,
        "output_path": model_output_path
    },
    datasets={"train": train_ds, "val": val_ds},
    scaling_config=scaling_config,
    run_config=RunConfig(storage_path="/mnt/shared_data/finflow/ray_results_multi")
)

trainer.fit()
