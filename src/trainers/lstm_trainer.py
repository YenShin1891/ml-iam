"""LSTM trainer with PyTorch Lightning, following TFT patterns."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, TQDMProgressBar
from sklearn.model_selection import ParameterSampler
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from configs.paths import RESULTS_PATH
from configs.models import LSTMTrainerConfig, LSTMSearchSpace


class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM sequences with proper group handling like TFT."""

    def __init__(
        self,
        data: pd.DataFrame,
        features: List[str],
        targets: List[str],
        time_idx: str = "Step",
        group_ids: List[str] = None,
        sequence_length: int = 1,
        mask_value: float = -1.0,
        scaler_X: Optional[StandardScaler] = None,
        scaler_y: Optional[StandardScaler] = None,
        fit_scalers: bool = True
    ):
        from configs.data import INDEX_COLUMNS

        self.sequence_length = sequence_length
        self.mask_value = mask_value
        self.features = features
        self.targets = targets
        self.time_idx = time_idx
        self.group_ids = group_ids if group_ids is not None else INDEX_COLUMNS

        # Extract features and targets
        X = data[features].copy()
        y = data[targets].values.copy()

        # Handle categorical columns first (before NaN filling)
        from configs.data import CATEGORICAL_COLUMNS
        categorical_cols_in_features = [col for col in CATEGORICAL_COLUMNS if col in X.columns]

        # Encode categorical columns like XGB does
        for col in categorical_cols_in_features:
            X[col] = X[col].astype('category').cat.codes

        # Handle NaN values
        X_filled = X.fillna(mask_value).astype(np.float32)

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Initialize scalers if not provided
        if scaler_X is None:
            scaler_X = StandardScaler()
        if scaler_y is None:
            scaler_y = StandardScaler()

        # Fit and transform or just transform
        if fit_scalers:
            self.X_scaled = scaler_X.fit_transform(X_filled)
            self.y_scaled = scaler_y.fit_transform(y)
        else:
            self.X_scaled = scaler_X.transform(X_filled)
            self.y_scaled = scaler_y.transform(y)

        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        # Create sequences WITHIN groups (like TFT)
        self.X_sequences = []
        self.y_sequences = []
        self.previous_targets = []  # Previous targets for teacher forcing
        self.masks = []
        self.group_info = []

        # Group by the group_ids columns
        for group_name, group_data in data.groupby(self.group_ids):
            group_indices = group_data.index
            group_size = len(group_data)

            # Create sequences within this group only
            for i in range(group_size - sequence_length + 1):
                # Get global indices for this group
                start_idx = group_indices[i]
                end_idx = group_indices[i + sequence_length - 1]

                # Find positions in scaled arrays
                start_pos = data.index.get_loc(start_idx)
                end_pos = data.index.get_loc(end_idx)

                # Extract sequence from scaled data
                x_seq = self.X_scaled[start_pos:start_pos + sequence_length]
                y_seq = self.y_scaled[end_pos]  # Predict the last item in sequence

                # Create previous targets for teacher forcing
                # Previous targets are the ground truth targets from the sequence
                if start_pos > 0:
                    # Get previous targets for the entire sequence
                    prev_targets_seq = self.y_scaled[start_pos-1:start_pos+sequence_length-1]  # Shifted by 1
                else:
                    # For the first sequence, pad with zeros
                    prev_targets_seq = np.zeros((sequence_length, self.y_scaled.shape[1]))
                    if sequence_length > 1:
                        prev_targets_seq[1:] = self.y_scaled[start_pos:start_pos+sequence_length-1]

                # Create mask for padded values
                mask_data = X_filled.iloc[start_pos:start_pos + sequence_length]
                mask = (mask_data != mask_value).all(axis=1)

                self.X_sequences.append(torch.FloatTensor(x_seq))
                self.y_sequences.append(torch.FloatTensor(y_seq))
                self.previous_targets.append(torch.FloatTensor(prev_targets_seq))
                self.masks.append(torch.FloatTensor(mask.values))
                self.group_info.append(group_name)

        self.X_sequences = torch.stack(self.X_sequences)
        self.y_sequences = torch.stack(self.y_sequences)
        self.previous_targets = torch.stack(self.previous_targets)
        self.masks = torch.stack(self.masks)

    def __len__(self):
        return len(self.X_sequences)

    def __getitem__(self, idx):
        return {
            'x': self.X_sequences[idx],
            'y': self.y_sequences[idx],
            'previous_targets': self.previous_targets[idx],  # For teacher forcing
            'mask': self.masks[idx],
            'group': self.group_info[idx]
        }


class LSTMModel(LightningModule):
    """PyTorch Lightning LSTM model."""

    def __init__(
        self,
        exogenous_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        include_previous_target: bool = True,  # Whether to include previous target as input
        dropout: float = 0.0,
        bidirectional: bool = False,
        dense_hidden_size: int = 64,
        dense_dropout: float = 0.0,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        scheduler: Optional[str] = None,
        scheduler_params: Optional[Dict] = None,
        mask_value: float = -1.0
    ):
        super().__init__()
        self.save_hyperparameters()

        self.exogenous_size = exogenous_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.include_previous_target = include_previous_target
        self.learning_rate = learning_rate
        self.mask_value = mask_value

        # LSTM layer - input size = exogenous (u_t) + (previous target y_{t-1} if enabled)
        lstm_input_size = exogenous_size + (output_size if include_previous_target else 0)
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Determine LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Dense layers for target prediction
        self.dense = nn.Sequential(
            nn.Linear(lstm_output_size, dense_hidden_size),
            nn.ReLU(),
            nn.Dropout(dense_dropout),
            nn.Linear(dense_hidden_size, output_size)
        )

        self.optimizer_name = optimizer
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params or {}

    def forward(self, exogenous_seq, mask=None, previous_targets=None, teacher_forcing=True):
        """
        Step-by-step LSTM unrolling with proper teacher forcing.

        At each timestep t, input is: v_t = [u_t, y_{t-1}]
        - u_t: ALL exogenous features observed at timestep t (always from dataset)
        - y_{t-1}: previous target (teacher forcing vs autoregressive)

        Args:
            exogenous_seq: ALL features [batch_size, seq_len, exogenous_size] - always from dataset
            mask: Optional mask for variable-length sequences
            previous_targets: Previous targets for teacher forcing [batch_size, seq_len, output_size]
            teacher_forcing: Whether to use teacher forcing (True) or autoregressive (False)
        """
        batch_size, seq_len, _ = exogenous_seq.size()

        if not self.include_previous_target:
            # Simple case: no previous targets, just use exogenous features
            x = exogenous_seq

            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).expand_as(x)
                x = x * mask_expanded

            lstm_out, _ = self.lstm(x)

            if mask is not None:
                valid_lengths = (~mask).sum(dim=1) - 1
                batch_indices = torch.arange(batch_size, device=x.device)
                last_output = lstm_out[batch_indices, valid_lengths]
            else:
                last_output = lstm_out[:, -1, :]

            return self.dense(last_output)

        # Step-by-step unrolling with teacher forcing
        hidden = None
        cell = None
        predictions = []

        # Initialize previous target (zeros for first timestep)
        prev_target = torch.zeros(batch_size, self.output_size, device=exogenous_seq.device, dtype=exogenous_seq.dtype)

        for t in range(seq_len):
            # Current exogenous features u_t (ALL features, always from dataset)
            u_t = exogenous_seq[:, t, :]  # [batch_size, exogenous_size]

            # Previous target y_{t-1} (teacher forcing or autoregressive)
            if t == 0:
                y_prev = prev_target  # First timestep uses zero
            else:
                if teacher_forcing and previous_targets is not None:
                    y_prev = previous_targets[:, t-1, :]  # Ground truth y_{t-1}
                else:
                    y_prev = prev_target  # Predicted y_{t-1}

            # Construct input: v_t = [u_t, y_{t-1}] - exactly as specified
            v_t = torch.cat([u_t, y_prev], dim=-1)  # [batch_size, exogenous_size + output_size]
            v_t = v_t.unsqueeze(1)  # [batch_size, 1, input_size]

            # Apply mask if needed
            if mask is not None and not mask[:, t].all():
                # Skip masked timesteps - keep previous prediction
                current_pred = prev_target
            else:
                # LSTM forward step
                lstm_out, (hidden, cell) = self.lstm(v_t, (hidden, cell) if hidden is not None else None)
                current_pred = self.dense(lstm_out[:, 0, :])  # [batch_size, output_size]

            predictions.append(current_pred)
            prev_target = current_pred  # Update for next timestep

        # Return only the last prediction (or all predictions if needed)
        return predictions[-1]  # [batch_size, output_size]

    def training_step(self, batch, batch_idx):
        x, y, mask = batch['x'], batch['y'], batch['mask']

        # x contains ALL features (u_t) - no splitting needed
        exogenous_seq = x

        # Get previous targets for teacher forcing if available
        previous_targets = batch.get('previous_targets', None)

        if self.include_previous_target and previous_targets is not None:
            # Use teacher forcing with ground truth previous targets
            y_hat = self(exogenous_seq, mask, previous_targets=previous_targets, teacher_forcing=True)
        else:
            # Standard forward pass without teacher forcing
            y_hat = self(exogenous_seq, mask, teacher_forcing=False)

        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch['x'], batch['y'], batch['mask']

        # x contains ALL features (u_t) - no splitting needed
        exogenous_seq = x

        # For validation, use autoregressive (no teacher forcing)
        y_hat = self(exogenous_seq, mask, teacher_forcing=False)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch['x'], batch['y'], batch['mask']

        # x contains ALL features (u_t) - no splitting needed
        exogenous_seq = x

        y_hat = self(exogenous_seq, mask, teacher_forcing=False)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return y_hat

    def predict_step(self, batch, batch_idx):
        x, mask = batch['x'], batch['mask']

        # x contains ALL features (u_t) - no splitting needed
        exogenous_seq = x

        return self(exogenous_seq, mask, teacher_forcing=False)

    def configure_optimizers(self):
        if self.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.scheduler_params.get("momentum", 0.9),
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        if self.scheduler_name is None:
            return optimizer

        if self.scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_params.get("T_max", 100)
            )
        elif self.scheduler_name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_params.get("step_size", 30),
                gamma=self.scheduler_params.get("gamma", 0.1)
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.scheduler_name}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        }


def create_lstm_datasets(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    features: List[str],
    targets: List[str],
    sequence_length: int = 1,
    mask_value: float = -1.0
) -> Tuple[LSTMDataset, LSTMDataset]:
    """Create LSTM datasets for training and validation with proper group handling."""

    train_dataset = LSTMDataset(
        train_data, features, targets,
        sequence_length=sequence_length,
        mask_value=mask_value,
        fit_scalers=True
    )

    val_dataset = LSTMDataset(
        val_data, features, targets,
        sequence_length=sequence_length,
        mask_value=mask_value,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        fit_scalers=False
    )

    return train_dataset, val_dataset


def create_lstm_dataloaders(
    train_dataset: LSTMDataset,
    val_dataset: LSTMDataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for LSTM training."""

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


def create_lstm_model(
    features: List[str],
    output_size: int,
    config: LSTMTrainerConfig,
    include_previous_target: bool = True
) -> LSTMModel:
    """Create LSTM model with given configuration."""

    # ALL features are exogenous (u_t)
    from configs.models.lstm import LSTMDatasetConfig
    dataset_config = LSTMDatasetConfig()
    feature_groups = dataset_config.build_feature_groups(features)

    exogenous_size = len(feature_groups["exogenous_features"])  # All features

    return LSTMModel(
        exogenous_size=exogenous_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=output_size,
        include_previous_target=include_previous_target,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        dense_hidden_size=config.dense_hidden_size,
        dense_dropout=config.dense_dropout,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        scheduler=config.scheduler,
        scheduler_params=config.scheduler_params,
        mask_value=config.mask_value
    )


def create_lstm_search_trainer(
    config: LSTMTrainerConfig,
    checkpoint_callback: ModelCheckpoint
) -> Trainer:
    """Create trainer for LSTM hyperparameter search with multi-device support."""
    early_stop = EarlyStopping(monitor=config.monitor, patience=config.patience, mode=config.mode)

    return Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices,  # Multi-device for search to handle 8 processes per GPU
        strategy="auto",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[early_stop, checkpoint_callback],
        logger=False,
        enable_progress_bar=False,
    )


def create_lstm_final_trainer(config: LSTMTrainerConfig) -> Trainer:
    """Create trainer for final LSTM model training."""
    # Custom progress bar with less frequent updates
    progress_bar = TQDMProgressBar(refresh_rate=5000)

    return Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,  # Use single device for final training to avoid distributed issues
        strategy="auto",
        gradient_clip_val=config.gradient_clip_val,
        logger=True,
        callbacks=[progress_bar],
        enable_checkpointing=False,  # Manual saving
        num_sanity_val_steps=0,  # Skip validation sanity checks
    )



def create_lstm_datasets_with_forecasting(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    features: List[str],
    targets: List[str],
    sequence_length: int = 10,
    mode: str = "train"
) -> tuple:
    """Create LSTM datasets with proper future feature handling for forecasting."""
    from configs.models.lstm import LSTMDatasetConfig

    dataset_config = LSTMDatasetConfig()
    feature_groups = dataset_config.build_feature_groups(features)

    known_future = feature_groups["time_varying_known_reals"] + feature_groups["time_varying_known_categoricals"]
    unknown_future = feature_groups["time_varying_unknown_reals"]

    if mode == "train":
        # Training: use all features as-is
        train_dataset = create_lstm_dataset(
            train_data, features, targets,
            sequence_length=sequence_length,
            mode="train",
            fit_scalers=True
        )
        val_dataset = create_lstm_dataset(
            val_data, features, targets,
            sequence_length=sequence_length,
            mode="val",
            scalers=train_dataset.scalers
        )
        return train_dataset, val_dataset, len(unknown_future)

    elif mode == "predict":
        # Prediction: limit future unknown features to historical values only
        # Create modified test data for proper forecasting
        predict_data = test_data.copy()

        # For unknown_future features: use only historical values (lag them forward)
        # For known_future features: use actual future values
        for col in unknown_future:
            if col in predict_data.columns:
                # Use last known value for each group (simple forward fill)
                predict_data[col] = predict_data.groupby(['Model', 'Scenario', 'Region'])[col].ffill()

        test_dataset = create_lstm_dataset(
            predict_data, features, targets,
            sequence_length=sequence_length,
            mode="predict"
        )
        return test_dataset, len(unknown_future)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def hyperparameter_search_lstm_parallel(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    targets: List[str],
    run_id: str,
    features: List[str] = None
) -> Dict:
    """Perform parallel hyperparameter search for LSTM model."""
    import torch.multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    import torch.distributed as dist

    search_cfg = LSTMSearchSpace()
    if features is None:
        from configs.data import NON_FEATURE_COLUMNS, INDEX_COLUMNS
        exclude_columns = NON_FEATURE_COLUMNS + INDEX_COLUMNS + ['Step']
        features = [col for col in train_data.columns if col not in targets and col not in exclude_columns]

    # Generate all parameter combinations
    param_list = list(ParameterSampler(
        search_cfg.param_dist, n_iter=search_cfg.search_iter_n, random_state=0
    ))

    # Get number of available GPUs
    num_gpus = torch.cuda.device_count()
    logging.info(f"Using {num_gpus} GPUs for parallel hyperparameter search")

    # Group parameters by GPU (distribute evenly)
    param_groups = [[] for _ in range(num_gpus)]
    for i, params in enumerate(param_list):
        param_groups[i % num_gpus].append((i, params))

    # Function to train on a specific GPU
    def train_on_gpu(gpu_id, param_assignments, shared_results):
        torch.cuda.set_device(gpu_id)
        device_results = []

        for trial_id, params in param_assignments:
            logging.info(f"GPU {gpu_id} - Trial {trial_id+1}: {params}")

            try:
                # Create datasets
                sequence_length = params.get("sequence_length", 1)
                train_dataset, val_dataset = create_lstm_datasets(
                    train_data, val_data, features, targets, sequence_length=sequence_length
                )

                # Create data loaders
                batch_size = params.get("batch_size", 32)
                train_loader, val_loader = create_lstm_dataloaders(
                    train_dataset, val_dataset, batch_size=batch_size
                )

                # Create model with search parameters
                config = LSTMTrainerConfig(
                    hidden_size=params.get("hidden_size", 64),
                    num_layers=params.get("num_layers", 1),
                    dropout=params.get("dropout", 0.0),
                    bidirectional=params.get("bidirectional", False),
                    dense_hidden_size=params.get("dense_hidden_size", 64),
                    dense_dropout=params.get("dense_dropout", 0.0),
                    learning_rate=params.get("learning_rate", 0.001),
                    batch_size=batch_size,
                    weight_decay=params.get("weight_decay", 0.0),
                    sequence_length=sequence_length,
                    max_epochs=20,
                    patience=3,
                    devices=1  # Single device per process
                )

                output_size = len(targets)
                model = create_lstm_model(features, output_size, config)

                # Create trainer for single GPU
                search_checkpoint = ModelCheckpoint(
                    dirpath=os.path.join(RESULTS_PATH, run_id, "search", f"trial_{trial_id}"),
                    filename="best",
                    monitor="val_loss",
                    mode="min",
                    save_top_k=1
                )

                trainer = Trainer(
                    max_epochs=config.max_epochs,
                    accelerator="gpu",
                    devices=[gpu_id],  # Specific GPU
                    strategy="auto",
                    gradient_clip_val=config.gradient_clip_val,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=config.patience, mode="min"), search_checkpoint],
                    logger=False,
                    enable_progress_bar=False,
                )

                # Train
                trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

                # Get validation loss
                val_loss = trainer.callback_metrics["val_loss"].item()
                result = {**params, "val_loss": val_loss, "trial_id": trial_id}
                device_results.append(result)

                logging.info(f"GPU {gpu_id} - Trial {trial_id+1} completed with val_loss: {val_loss:.4f}")

            except Exception as e:
                logging.error(f"GPU {gpu_id} - Trial {trial_id+1} failed: {str(e)}")
                device_results.append({**params, "val_loss": float("inf"), "trial_id": trial_id, "error": str(e)})

        shared_results.extend(device_results)

    # Run parallel training
    with mp.Manager() as manager:
        shared_results = manager.list()
        processes = []

        for gpu_id in range(num_gpus):
            if param_groups[gpu_id]:  # Only start process if there are parameters to train
                p = mp.Process(target=train_on_gpu, args=(gpu_id, param_groups[gpu_id], shared_results))
                p.start()
                processes.append(p)

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Convert to regular list
        search_results = list(shared_results)

    # Find best parameters
    valid_results = [r for r in search_results if r["val_loss"] != float("inf")]
    if not valid_results:
        raise RuntimeError("All hyperparameter trials failed")

    best_result = min(valid_results, key=lambda x: x["val_loss"])
    best_params = {k: v for k, v in best_result.items() if k not in ["val_loss", "trial_id", "error"]}
    best_score = best_result["val_loss"]

    # Save search results
    search_results_df = pd.DataFrame(search_results)
    search_results_path = os.path.join(RESULTS_PATH, run_id, "search_results.csv")
    os.makedirs(os.path.dirname(search_results_path), exist_ok=True)
    search_results_df.to_csv(search_results_path, index=False)
    logging.info(f"Search results saved to: {search_results_path}")

    logging.info(f"Best LSTM Params: {best_params} with Val Loss: {best_score:.4f}")
    return best_params


def hyperparameter_search_lstm(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    targets: List[str],
    run_id: str,
    features: List[str] = None
) -> Dict:
    """Perform hyperparameter search for LSTM model (wrapper function)."""
    # Use parallel search if multiple GPUs available, otherwise sequential
    if torch.cuda.device_count() > 1:
        logging.info(f"Using parallel hyperparameter search with {torch.cuda.device_count()} GPUs")
        return hyperparameter_search_lstm_parallel(train_data, val_data, targets, run_id, features)
    else:
        logging.info("Using sequential hyperparameter search (single GPU)")
        return hyperparameter_search_lstm_sequential(train_data, val_data, targets, run_id, features)


def hyperparameter_search_lstm_sequential(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    targets: List[str],
    run_id: str,
    features: List[str] = None
) -> Dict:
    """Sequential hyperparameter search (original implementation)."""

    search_cfg = LSTMSearchSpace()
    # Use features from preprocessing if provided, otherwise derive from columns (but this shouldn't happen)
    if features is None:
        from configs.data import NON_FEATURE_COLUMNS, INDEX_COLUMNS
        exclude_columns = NON_FEATURE_COLUMNS + INDEX_COLUMNS + ['Step']  # Step is added by sequence preprocessing
        features = [col for col in train_data.columns if col not in targets and col not in exclude_columns]

    search_results = []
    best_score = float("inf")
    best_params = None

    for i, params in enumerate(ParameterSampler(
        search_cfg.param_dist, n_iter=search_cfg.search_iter_n, random_state=0
    )):
        logging.info(f"LSTM Search Iteration {i+1}/{search_cfg.search_iter_n} - Params: {params}")

        # Create datasets
        sequence_length = params.get("sequence_length", 1)
        train_dataset, val_dataset = create_lstm_datasets(
            train_data, val_data, features, targets, sequence_length=sequence_length
        )

        # Create data loaders
        batch_size = params.get("batch_size", 32)
        train_loader, val_loader = create_lstm_dataloaders(
            train_dataset, val_dataset, batch_size=batch_size
        )

        # Create model with search parameters
        config = LSTMTrainerConfig(
            hidden_size=params.get("hidden_size", 64),
            num_layers=params.get("num_layers", 1),
            dropout=params.get("dropout", 0.0),
            bidirectional=params.get("bidirectional", False),
            dense_hidden_size=params.get("dense_hidden_size", 64),
            dense_dropout=params.get("dense_dropout", 0.0),
            learning_rate=params.get("learning_rate", 0.001),
            batch_size=batch_size,
            weight_decay=params.get("weight_decay", 0.0),
            sequence_length=sequence_length,
            max_epochs=20,  # Reduced for search
            patience=3,  # Reduced for search
            devices=1  # Use single device for sequential search
        )

        output_size = len(targets)

        model = create_lstm_model(features, output_size, config)

        # Create trainer for search
        search_checkpoint = ModelCheckpoint(
            dirpath=os.path.join(RESULTS_PATH, run_id, "search", f"trial_{i}"),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )

        trainer = create_lstm_search_trainer(config, search_checkpoint)

        # Train
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Get validation loss
        val_loss = trainer.callback_metrics["val_loss"].item()
        search_results.append({**params, "val_loss": val_loss})

        if val_loss < best_score:
            best_score = val_loss
            best_params = params

        logging.info(f"Trial {i+1} Val Loss: {val_loss:.4f}")

    # Save search results like TFT
    search_results_df = pd.DataFrame(search_results)
    search_results_path = os.path.join(RESULTS_PATH, run_id, "search_results.csv")
    os.makedirs(os.path.dirname(search_results_path), exist_ok=True)
    search_results_df.to_csv(search_results_path, index=False)
    logging.info(f"Search results saved to: {search_results_path}")

    logging.info(f"Best LSTM Params: {best_params} with Val Loss: {best_score:.4f}")
    return best_params


def train_final_lstm(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    targets: List[str],
    run_id: str,
    best_params: Dict,
    session_state: Optional[Dict] = None,
    features: List[str] = None
) -> None:
    """Train final LSTM model with best parameters."""

    # Combine train and validation data (like TFT's create_combined_dataset)
    combined_data = pd.concat([train_data, val_data], ignore_index=True)
    # Use features from preprocessing if provided, otherwise derive from columns (but this shouldn't happen)
    if features is None:
        from configs.data import NON_FEATURE_COLUMNS, INDEX_COLUMNS
        exclude_columns = NON_FEATURE_COLUMNS + INDEX_COLUMNS + ['Step']  # Step is added by sequence preprocessing
        features = [col for col in train_data.columns if col not in targets and col not in exclude_columns]

    # Create dataset with combined data
    sequence_length = best_params.get("sequence_length", 1)
    combined_dataset = LSTMDataset(
        combined_data, features, targets, sequence_length=sequence_length, fit_scalers=True
    )

    # Create data loader
    batch_size = best_params.get("batch_size", 32)
    combined_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    # Create final model
    config = LSTMTrainerConfig(
        hidden_size=best_params.get("hidden_size", 64),
        num_layers=best_params.get("num_layers", 1),
        dropout=best_params.get("dropout", 0.0),
        bidirectional=best_params.get("bidirectional", False),
        dense_hidden_size=best_params.get("dense_hidden_size", 64),
        dense_dropout=best_params.get("dense_dropout", 0.0),
        learning_rate=best_params.get("learning_rate", 0.001),
        batch_size=batch_size,
        weight_decay=best_params.get("weight_decay", 0.0),
        sequence_length=sequence_length
    )

    output_size = len(targets)

    model = create_lstm_model(features, output_size, config)

    # Create final trainer
    final_dir = os.path.join(RESULTS_PATH, run_id, "final")
    os.makedirs(final_dir, exist_ok=True)

    trainer = create_lstm_final_trainer(config)

    # Train on combined data
    trainer.fit(model=model, train_dataloaders=combined_loader)

    # Save final checkpoint manually
    final_ckpt_path = os.path.join(final_dir, "best.ckpt")
    trainer.save_checkpoint(final_ckpt_path)

    # Save scalers and config to session state
    if session_state is not None:
        session_state["lstm_scaler_X"] = combined_dataset.scaler_X
        session_state["lstm_scaler_y"] = combined_dataset.scaler_y
        session_state["lstm_config"] = config
        session_state["lstm_sequence_length"] = sequence_length


def predict_lstm(session_state: Dict, run_id: str) -> np.ndarray:
    """Make predictions using trained LSTM model."""
    from src.trainers.evaluation import save_metrics

    # Get data from session state (stored as DataFrames like TFT)
    test_data = session_state["test_data"]
    targets = session_state["targets"]
    features = session_state["features"]  # Use features from preprocessing, just like TFT


    # Load model
    final_ckpt_path = os.path.join(RESULTS_PATH, run_id, "final", "best.ckpt")

    if not os.path.exists(final_ckpt_path):
        raise FileNotFoundError(f"Final model checkpoint not found: {final_ckpt_path}")

    # Get config and scalers from session state
    config = session_state.get("lstm_config")
    scaler_X = session_state.get("lstm_scaler_X")
    scaler_y = session_state.get("lstm_scaler_y")
    sequence_length = session_state.get("lstm_sequence_length", 1)

    if config is None or scaler_X is None or scaler_y is None:
        raise ValueError("LSTM config and scalers not found in session state")

    # Load model
    model = LSTMModel.load_from_checkpoint(final_ckpt_path)
    model.eval()

    # Create test dataset - autoregressive prediction during inference
    # Note: Categorical encoding happens inside LSTMDataset constructor
    test_dataset = LSTMDataset(
        test_data, features, targets,
        sequence_length=sequence_length,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        fit_scalers=False
    )

    # Create data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=False
    )

    # Create trainer for prediction
    trainer = Trainer(
        devices=1,  # Use single device for prediction to avoid distributed issues
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=False,  # Disable progress bar to reduce log clutter
        logger=False
    )

    # Make predictions
    raw_predictions = trainer.predict(model, test_loader)
    predictions_list = [p.cpu().numpy() for p in raw_predictions]
    predictions_array = np.vstack(predictions_list)

    # Post-process predictions (like TFT)
    if predictions_array.ndim == 3 and predictions_array.shape[1] == 1:
        predictions_array = predictions_array.squeeze(axis=1)

    logging.info("Raw predictions shape: %s", predictions_array.shape)

    # Inverse transform predictions
    predictions_unscaled = scaler_y.inverse_transform(predictions_array)

    # Align predictions with test data properly
    # LSTM predictions correspond to sequences within groups, not direct row mapping
    aligned_preds = np.full((len(test_data), len(targets)), np.nan)

    # Process each group separately to handle sequence alignment
    pred_idx = 0
    from configs.data import INDEX_COLUMNS
    group_ids = INDEX_COLUMNS

    for group_name, group_data in test_data.groupby(group_ids):
        group_size = len(group_data)
        group_indices = group_data.index

        # Number of valid sequences in this group
        num_sequences = max(0, group_size - sequence_length + 1)

        # Align predictions for this group
        for i in range(num_sequences):
            if pred_idx < len(predictions_unscaled):
                # The prediction corresponds to the last item of the sequence
                target_row_idx = group_indices[i + sequence_length - 1]
                data_row_idx = test_data.index.get_loc(target_row_idx)

                aligned_preds[data_row_idx] = predictions_unscaled[pred_idx]
                pred_idx += 1

    logging.info("LSTM prediction completed. Shape: %s", aligned_preds.shape)

    # Get valid (non-NaN) predictions and targets for metrics (like TFT pattern)
    y_test = test_data[targets].values
    valid_mask = ~np.isnan(aligned_preds).any(axis=1)
    valid_preds = aligned_preds[valid_mask]
    valid_targets = y_test[valid_mask]

    logging.info("Valid predictions: %d out of %d", len(valid_preds), len(y_test))

    # Save metrics using only valid predictions and targets
    save_metrics(run_id, valid_targets, valid_preds)

    # Store horizon data for plotting (like TFT pattern)
    session_state["horizon_df"] = test_data
    session_state["horizon_y_true"] = valid_targets

    return aligned_preds


__all__ = [
    "hyperparameter_search_lstm",
    "train_final_lstm",
    "predict_lstm",
    "create_lstm_datasets",
    "create_lstm_dataloaders",
    "create_lstm_model",
    "create_lstm_search_trainer",
    "create_lstm_final_trainer",
]