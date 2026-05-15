import torch
import torch.nn as nn
import os
import ray
from ray import tune
from ray.train import CheckpointConfig
from ray.tune import RunConfig
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import ASHAScheduler

# ---> NEW: Import W&B Logger
from ray.air.integrations.wandb import WandbLoggerCallback 

os.environ.pop("RAY_AIR_VERBOSITY", None)
os.environ.pop("RAY_TUNE_VERBOSITY", None)

ray.init(address="auto", namespace="OBIS_TEST", ignore_reinit_error=True)

train_dir = '/mnt/shared_data/finflow/precomputed_sequences_multi_train'
val_dir = '/mnt/shared_data/finflow/precomputed_sequences_multi_val'

print("Loading precomputed multivariate datasets for tuning...")
train_ds = ray.data.read_parquet(train_dir)
val_ds = ray.data.read_parquet(val_dir)

def train_loop(config, train_data, val_data):
    class LSTMModel(nn.Module):
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

    model = LSTMModel(
        input_size=3, 
        hidden_size=config["hidden_size"], 
        num_layers=config["num_layers"], 
        dropout=config["dropout"]
    )
    model = torch.compile(model)
    model.to("cuda") 

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    batch_size = config["batch_size"]

    for epoch in range(50):
        model.train()
        train_loss, train_batches = 0, 0

        for batch in train_data.iter_torch_batches(batch_size=batch_size, local_shuffle_buffer_size=10000):
            X_batch = batch["X"].float().view(-1, 12, 3).to("cuda")
            y_batch = batch["y"].float().to("cuda")

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

        model.eval()
        val_loss, val_batches = 0, 0
        with torch.no_grad():
            for batch in val_data.iter_torch_batches(batch_size=batch_size):
                X_val_batch = batch["X"].float().view(-1, 12, 3).to("cuda")
                y_val_batch = batch["y"].float().to("cuda")

                X_val_batch = torch.log1p(X_val_batch)
                y_val_batch = torch.log1p(y_val_batch)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    val_pred = model(X_val_batch)
                    batch_loss = loss_fn(val_pred, y_val_batch)

                val_loss += batch_loss.item()
                val_batches += 1

        val_loss /= max(val_batches, 1)

        # ---> BUG FIX: Using tune.report
        tune.report({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch + 1})


print("Setting up Bayesian Optimization with Optuna and ASHA...")

search_space = {
    "hidden_size": tune.choice([64, 128, 256]),
    "num_layers": tune.choice([1, 2, 3]),
    "lr": tune.loguniform(1e-4, 1e-2),
    "dropout": tune.uniform(0.1, 0.4),
    "batch_size": tune.choice([512, 1024, 2048])
}

asha_scheduler = ASHAScheduler(
    time_attr="training_iteration",
    max_t=50,
    grace_period=5,
    reduction_factor=2
)

trainable_with_resources = tune.with_resources(
    tune.with_parameters(train_loop, train_data=train_ds, val_data=val_ds),
    resources={"cpu": 2, "gpu": 1} 
)

cc = CheckpointConfig()
try:
    cc.checkpoint_at_end = False  
    cc.checkpoint_frequency = 0   
except Exception:
    pass

run_config = RunConfig(
    name="lstm_bayes_opt_v2",
    storage_path="/mnt/shared_data/finflow/ray_tune_results",
    checkpoint_config=cc,
    # ---> NEW: Activate the W&B Callback
    callbacks=[
        WandbLoggerCallback(
            project="finflow-lstm-tuning",
            api_key=os.environ.get("WANDB_API_KEY", "PASTE_YOUR_API_KEY_HERE_IF_ENV_FAILS") 
        )
    ]
)

import ray.tune.experimental.output
import ray.tune.tune

def safe_get_air_verbosity(verbose):
    if isinstance(verbose, int): return verbose
    if hasattr(verbose, "value"): return verbose.value
    if isinstance(verbose, str): return int(verbose) if verbose.isdigit() else 2
    return 2

ray.tune.experimental.output.get_air_verbosity = safe_get_air_verbosity
ray.tune.tune.get_air_verbosity = safe_get_air_verbosity

EXPERIMENT_DIR = "/mnt/shared_data/finflow/ray_tune_results/lstm_bayes_opt_v2"

tuner = tune.Tuner.restore(
    path=EXPERIMENT_DIR,
    trainable=trainable_with_resources, # Matches your original setup
    resume_unfinished=True,             # Picks up pending trials
    resume_errored=True                 # Restarts the ones that crashed
)

results = tuner.fit()
best_result = results.get_best_result("val_loss", "min")

print("\n TUNING COMPLETE!")
print(f"Best validation loss: {best_result.metrics['val_loss']}")
print(f"Best hyperparameters: {best_result.config}")

best_history_df = best_result.metrics_dataframe
clean_history = best_history_df[['training_iteration', 'train_loss', 'val_loss']]
clean_history = clean_history.rename(columns={'training_iteration': 'epoch'})
clean_history.to_csv('/mnt/shared_data/finflow/lstm_models_multi/best_tuned_history.csv', index=False)
print("Saved the champion model's loss history to best_tuned_history.csv!")
