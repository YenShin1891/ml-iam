"""LSTM trainer with PyTorch Lightning, following TFT patterns."""

from dataclasses import dataclass
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

from src.utils.utils import get_run_root, is_primary_rank
from src.trainers.progress import EpochProgressLogger
from configs.data import MAX_CONTEXT_LENGTH
from configs.models import LSTMTrainerConfig, LSTMSearchSpace
from .search import (
    completed_trials,
    params_signature,
    select_top_k_signatures,
    write_search_report,
)


def _infer_non_numeric_feature_columns(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Return feature columns that are not numeric in the provided dataframe."""
    non_numeric: List[str] = []
    for col in features:
        if col not in df.columns:
            # Missing columns will be created as NaN later via reindex; treat as numeric.
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    return non_numeric


def _one_hot_encode_and_align_features(
    train_data: pd.DataFrame,
    other_data: List[pd.DataFrame],
    features: List[str],
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List[str], List[str]]:
    """One-hot encode non-numeric feature columns based on train_data and align others.

    Returns:
      (encoded_train_data, encoded_other_data_list, encoded_feature_columns, non_numeric_feature_columns)
    """
    non_numeric_cols = _infer_non_numeric_feature_columns(train_data, features)
    if not non_numeric_cols:
        return train_data, other_data, features, []

    # Encode train
    X_train = train_data.reindex(columns=features)
    X_train_enc = pd.get_dummies(X_train, columns=non_numeric_cols, dummy_na=True)
    # Ensure stable float dtype for dummy columns; keep numeric columns as-is.
    for col in X_train_enc.columns:
        if X_train_enc[col].dtype == bool:
            X_train_enc[col] = X_train_enc[col].astype(np.float32)
        elif pd.api.types.is_integer_dtype(X_train_enc[col]):
            X_train_enc[col] = X_train_enc[col].astype(np.float32)

    encoded_features = list(X_train_enc.columns)
    encoded_train = pd.concat([train_data.drop(columns=features, errors="ignore"), X_train_enc], axis=1)

    encoded_others: List[pd.DataFrame] = []
    for df in other_data:
        X_other = df.reindex(columns=features)
        X_other_enc = pd.get_dummies(X_other, columns=non_numeric_cols, dummy_na=True)
        # Align to training columns (unknown categories become all-zeros)
        X_other_enc = X_other_enc.reindex(columns=encoded_features, fill_value=0.0)
        for col in X_other_enc.columns:
            if X_other_enc[col].dtype == bool:
                X_other_enc[col] = X_other_enc[col].astype(np.float32)
            elif pd.api.types.is_integer_dtype(X_other_enc[col]):
                X_other_enc[col] = X_other_enc[col].astype(np.float32)
        encoded_other = pd.concat([df.drop(columns=features, errors="ignore"), X_other_enc], axis=1)
        encoded_others.append(encoded_other)

    logging.warning(
        "Detected non-numeric LSTM features %s; applying one-hot encoding (%d -> %d columns)",
        non_numeric_cols,
        len(features),
        len(encoded_features),
    )

    return encoded_train, encoded_others, encoded_features, non_numeric_cols


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
        target_offset: int = 0,
        mask_value: float = -1.0,
        scaler_X: Optional[StandardScaler] = None,
        scaler_y: Optional[StandardScaler] = None,
        fit_scalers: bool = True,
        categorical_features: Optional[List[str]] = None,
    ):
        from configs.data import INDEX_COLUMNS
        from src.data.preprocess import observed_mask_columns, sanitize_target_scaler
        self.sequence_length = sequence_length
        self.target_offset = target_offset
        self.mask_value = mask_value
        self.features = features
        self.targets = targets
        self.time_idx = time_idx
        self.group_ids = group_ids if group_ids is not None else INDEX_COLUMNS
        self.categorical_features = categorical_features or []

        # Separate continuous features (to be scaled) from categorical (kept as int codes)
        continuous_features = [f for f in features if f not in self.categorical_features]

        # Extract observed-target mask (per element)
        obs_cols = observed_mask_columns(targets)
        if all(c in data.columns for c in obs_cols):
            self._obs_mask = data[obs_cols].values.astype(np.float32)
        else:
            self._obs_mask = np.ones((len(data), len(targets)), dtype=np.float32)

        # Extract continuous features and targets
        X_cont = data[continuous_features].copy() if continuous_features else pd.DataFrame(index=data.index)
        y = data[targets].values.copy()
        # Hide unobserved targets from the scaler: StandardScaler ignores NaN
        # when fitting, so its statistics describe real observations only.
        # The scaled array is zero-filled afterwards and the loss masks those
        # elements out via `target_obs`.
        y = np.where(self._obs_mask.astype(bool), y, np.nan)

        # Handle NaN values in continuous features
        X_cont_filled = X_cont.fillna(mask_value).astype(np.float32)

        # Extract categorical features as integer codes (no scaling)
        if self.categorical_features:
            X_cat = data[self.categorical_features].values.astype(np.int64)
        else:
            X_cat = np.empty((len(data), 0), dtype=np.int64)

        # Ensure y is 2D
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Initialize scalers if not provided
        if scaler_X is None:
            scaler_X = StandardScaler()
        if scaler_y is None:
            scaler_y = StandardScaler()

        # Fit and transform or just transform (only continuous features)
        if continuous_features:
            if fit_scalers:
                self.X_cont_scaled = scaler_X.fit_transform(X_cont_filled)
            else:
                self.X_cont_scaled = scaler_X.transform(X_cont_filled)
        else:
            self.X_cont_scaled = np.empty((len(data), 0), dtype=np.float32)

        if fit_scalers:
            self.y_scaled = scaler_y.fit_transform(y)
            if sanitize_target_scaler(scaler_y, targets):
                self.y_scaled = scaler_y.transform(y)
        else:
            self.y_scaled = scaler_y.transform(y)
        # Unobserved elements are NaN after scaling; zero them so the tensors
        # stay finite (the loss ignores them through `target_obs`).
        self.y_scaled = np.nan_to_num(self.y_scaled, nan=0.0)

        self.X_cat = X_cat
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        # Create sequences WITHIN groups (like TFT)
        self.X_sequences = []       # continuous features (scaled)
        self.cat_sequences = []     # categorical indices (unscaled)
        self.y_sequences = []
        self.target_obs = []        # per-target observed mask
        self.masks = []
        self.group_info = []
        self.target_positions = []  # row of `data` each sequence predicts

        # Group by the group_ids columns
        for group_name, group_data in data.groupby(self.group_ids):
            group_indices = group_data.index
            group_size = len(group_data)

            # Number of possible sequences respecting target_offset
            max_start = group_size - (sequence_length + target_offset) + 1
            if max_start < 1:
                # Not enough data in this group for a single sequence with given offset; skip
                continue

            # A short context fits more windows into the same series, and the
            # extra ones sit at the start, where the series is easiest.  Skip
            # them so every context length predicts the same target rows:
            # otherwise a comparison between context lengths measures which
            # rows each was scored on rather than how much history helped.
            first_start = max(0, MAX_CONTEXT_LENGTH - sequence_length)
            if first_start >= max_start:
                continue

            for i in range(first_start, max_start):
                start_idx = group_indices[i]
                target_idx = group_indices[i + sequence_length - 1 + target_offset]

                start_pos = data.index.get_loc(start_idx)
                target_pos = data.index.get_loc(target_idx)

                # Extract sequence (historical context)
                x_seq = self.X_cont_scaled[start_pos:start_pos + sequence_length]
                cat_seq = self.X_cat[start_pos:start_pos + sequence_length]
                # Target is offset ahead of sequence end
                y_seq = self.y_scaled[target_pos]

                # Create mask for padded values
                mask_data = X_cont_filled.iloc[start_pos:start_pos + sequence_length]
                mask = (mask_data != mask_value).all(axis=1) if len(continuous_features) > 0 else pd.Series(True, index=mask_data.index)

                self.X_sequences.append(torch.FloatTensor(x_seq))
                self.cat_sequences.append(torch.LongTensor(cat_seq))
                self.y_sequences.append(torch.FloatTensor(y_seq))
                self.target_obs.append(torch.FloatTensor(self._obs_mask[target_pos]))
                self.masks.append(torch.FloatTensor(mask.values if hasattr(mask, 'values') else [mask]))
                self.group_info.append(group_name)
                self.target_positions.append(target_pos)

        self.X_sequences = torch.stack(self.X_sequences)
        self.cat_sequences = torch.stack(self.cat_sequences)
        self.y_sequences = torch.stack(self.y_sequences)
        self.target_obs = torch.stack(self.target_obs)
        self.masks = torch.stack(self.masks)
        self.target_positions = np.asarray(self.target_positions, dtype=int)

    def __len__(self):
        return len(self.X_sequences)

    def __getitem__(self, idx):
        return {
            'x': self.X_sequences[idx],
            'cat': self.cat_sequences[idx],
            'y': self.y_sequences[idx],
            'target_obs': self.target_obs[idx],
            'mask': self.masks[idx],
            'group': self.group_info[idx]
        }


def align_sequence_predictions(dataset, predictions, n_rows: int) -> np.ndarray:
    """Scatter one prediction per sequence back onto the rows it belongs to.

    Rows the model produced nothing for — the first `sequence_length +
    target_offset - 1` of each group, and groups too short for a single
    sequence — stay NaN.

    LSTMDataset already recorded which row each sequence predicts, so read it
    rather than re-deriving the mapping from group sizes: a second
    implementation of the same arithmetic can drift from the dataset's without
    anything failing loudly, silently shifting every prediction.
    """
    predictions = np.asarray(predictions)
    positions = dataset.target_positions

    if len(positions) != len(predictions):
        raise ValueError(
            f"Model returned {len(predictions)} predictions for {len(positions)} sequences"
        )

    aligned = np.full((n_rows, predictions.shape[1]), np.nan)
    if len(positions):
        aligned[positions] = predictions
    return aligned


class LSTMModel(LightningModule):
    """PyTorch Lightning LSTM model."""

    def __init__(
        self,
        exogenous_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        dense_hidden_size: int = 64,
        dense_dropout: float = 0.0,
        learning_rate: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        scheduler: Optional[str] = None,
        scheduler_params: Optional[Dict] = None,
        mask_value: float = -1.0,
        target_offset: int = 1,
        num_model_families: int = 0,
        num_regions: int = 0,
        embedding_dim: int = 8,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.exogenous_size = exogenous_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.mask_value = mask_value
        self.target_offset = target_offset
        self.num_model_families = num_model_families
        self.num_regions = num_regions
        self.embedding_dim = embedding_dim

        # Categorical embeddings
        self.has_embeddings = num_model_families > 0 or num_regions > 0
        total_embedding_size = 0
        if num_model_families > 0:
            self.model_family_embedding = nn.Embedding(num_model_families, embedding_dim)
            total_embedding_size += embedding_dim
        if num_regions > 0:
            self.region_embedding = nn.Embedding(num_regions, embedding_dim)
            total_embedding_size += embedding_dim

        # LSTM layer — lag features are already part of exogenous_size
        lstm_input_size = exogenous_size + total_embedding_size
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

    def _embed_categoricals(self, cat_indices):
        """Compute embedding vectors from categorical indices.

        Column ordering follows CATEGORICAL_COLUMNS = ['Region', 'Model_Family']:
          col 0 = Region, col 1 = Model_Family

        Args:
            cat_indices: [batch_size, seq_len, num_cat_features] LongTensor

        Returns:
            [batch_size, seq_len, total_embedding_size] FloatTensor, or None if no embeddings.
        """
        if not self.has_embeddings:
            return None

        parts = []
        col = 0
        if self.num_regions > 0:
            parts.append(self.region_embedding(cat_indices[:, :, col]))
            col += 1
        if self.num_model_families > 0:
            parts.append(self.model_family_embedding(cat_indices[:, :, col]))
            col += 1
        return torch.cat(parts, dim=-1)

    def forward(self, exogenous_seq, mask=None, cat_indices=None, **_kwargs):
        """Batched LSTM forward pass.

        Args:
            exogenous_seq: Continuous features [batch_size, seq_len, exogenous_size]
            mask: Optional mask for variable-length sequences
            cat_indices: Categorical indices [batch_size, seq_len, num_cat_features]
        """
        batch_size, seq_len, _ = exogenous_seq.size()

        # Compute embeddings and concatenate with continuous features
        emb = self._embed_categoricals(cat_indices)
        if emb is not None:
            exogenous_seq = torch.cat([exogenous_seq, emb], dim=-1)

        x = exogenous_seq
        if mask is not None:
            x = x * mask.unsqueeze(-1).expand_as(x)

        lstm_out, (hidden, cell) = self.lstm(x)

        if mask is not None:
            valid_lengths = mask.sum(dim=1).long() - 1
            batch_indices = torch.arange(batch_size, device=x.device)
            last_output = lstm_out[batch_indices, valid_lengths]
        else:
            last_output = lstm_out[:, -1, :]

        if getattr(self, "target_offset", 0) and self.target_offset > 0:
            v_t = x[:, -1, :].unsqueeze(1)
            lstm_out2, _ = self.lstm(v_t, (hidden, cell))
            return self.dense(lstm_out2[:, 0, :])
        return self.dense(last_output)

    @staticmethod
    def _masked_mse(y_hat, y, target_obs):
        """MSE masked to originally-observed target elements only."""
        sq_err = (y_hat - y) ** 2
        return (sq_err * target_obs).sum() / target_obs.sum().clamp(min=1)

    def training_step(self, batch, batch_idx):
        x, y, mask = batch['x'], batch['y'], batch['mask']
        cat = batch.get('cat', None)
        target_obs = batch.get('target_obs', None)

        y_hat = self(x, mask, cat_indices=cat)
        if target_obs is not None:
            loss = self._masked_mse(y_hat, y, target_obs)
        else:
            loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, mask = batch['x'], batch['y'], batch['mask']
        cat = batch.get('cat', None)
        target_obs = batch.get('target_obs', None)

        y_hat = self(x, mask, cat_indices=cat)
        if target_obs is not None:
            loss = self._masked_mse(y_hat, y, target_obs)
        else:
            loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, mask = batch['x'], batch['y'], batch['mask']
        cat = batch.get('cat', None)
        target_obs = batch.get('target_obs', None)

        y_hat = self(x, mask, cat_indices=cat)
        if target_obs is not None:
            loss = self._masked_mse(y_hat, y, target_obs)
        else:
            loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss, sync_dist=True)
        return y_hat

    def predict_step(self, batch, batch_idx):
        x, mask = batch['x'], batch['mask']
        cat = batch.get('cat', None)

        return self(x, mask, cat_indices=cat)

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
    target_offset: int = 0,
    mask_value: float = -1.0,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[LSTMDataset, LSTMDataset]:
    """Create LSTM datasets for training and validation with proper group handling."""
    categorical_features = categorical_features or []
    continuous_features = [f for f in features if f not in categorical_features]

    # Ensure all continuous feature columns are numeric and aligned between train/val
    train_data_enc, (val_data_enc,), encoded_features, _ = _one_hot_encode_and_align_features(
        train_data,
        [val_data],
        continuous_features,
    )
    # Carry categorical columns through (already int-coded upstream)
    for col in categorical_features:
        if col in train_data.columns:
            train_data_enc[col] = train_data[col].values
        if col in val_data.columns:
            val_data_enc[col] = val_data[col].values

    # Full feature list = encoded continuous + categorical
    all_features = encoded_features + categorical_features

    train_dataset = LSTMDataset(
        train_data_enc, all_features, targets,
        sequence_length=sequence_length,
        target_offset=target_offset,
        mask_value=mask_value,
        fit_scalers=True,
        categorical_features=categorical_features,
    )

    val_dataset = LSTMDataset(
        val_data_enc, all_features, targets,
        sequence_length=sequence_length,
        target_offset=target_offset,
        mask_value=mask_value,
        scaler_X=train_dataset.scaler_X,
        scaler_y=train_dataset.scaler_y,
        fit_scalers=False,
        categorical_features=categorical_features,
    )

    return train_dataset, val_dataset, encoded_features


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
    num_model_families: int = 0,
    num_regions: int = 0,
) -> LSTMModel:
    """Create LSTM model with given configuration.

    Args:
        features: Continuous (encoded) feature names — categoricals are handled via embeddings.
    """
    # exogenous_size = number of continuous features in the data tensor
    exogenous_size = len(features)

    return LSTMModel(
        exogenous_size=exogenous_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        output_size=output_size,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
        dense_hidden_size=config.dense_hidden_size,
        dense_dropout=config.dense_dropout,
        learning_rate=config.learning_rate,
        optimizer=config.optimizer,
        weight_decay=config.weight_decay,
        scheduler=config.scheduler,
        scheduler_params=config.scheduler_params,
        mask_value=config.mask_value,
        target_offset=config.target_offset,
        num_model_families=num_model_families,
        num_regions=num_regions,
        embedding_dim=config.embedding_dim,
    )


def create_lstm_search_trainer(
    config: LSTMTrainerConfig,
    log_dir: Optional[str] = None,
) -> Trainer:
    """Create trainer for LSTM hyperparameter search with multi-device support.

    Trials only rank configurations, so nothing is checkpointed: the final
    fit retrains the winner from scratch and nothing ever read the per-trial
    weights.
    """
    early_stop = EarlyStopping(monitor=config.monitor, patience=config.patience, mode=config.mode)

    logger = False
    if log_dir:
        logger = CSVLogger(save_dir=log_dir, name="", version="")

    return Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices,  # Multi-device for search to handle 8 processes per GPU
        strategy="auto",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[early_stop],
        logger=logger,
        enable_progress_bar=False,
        enable_checkpointing=False,
    )


def create_lstm_final_trainer(
    config: LSTMTrainerConfig,
    ckpt_path: str,
    log_dir: Optional[str] = None,
) -> Trainer:
    """Create trainer for final LSTM model training with early stopping on val_loss."""
    early_stop = EarlyStopping(
        monitor="val_loss", patience=config.final_patience, mode="min",
    )
    checkpoint = ModelCheckpoint(
        dirpath=os.path.dirname(ckpt_path),
        filename=os.path.splitext(os.path.basename(ckpt_path))[0],
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        # A re-run train phase must overwrite best.ckpt: with the version
        # counter on, Lightning writes best-v1.ckpt instead and the test
        # phase keeps loading the stale weights.
        enable_version_counter=False,
    )
    progress = EpochProgressLogger("LSTM final training")

    logger = False
    if log_dir:
        logger = CSVLogger(save_dir=log_dir, name="", version="")

    return Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=config.devices,
        strategy="auto",
        gradient_clip_val=config.gradient_clip_val,
        callbacks=[early_stop, checkpoint, progress],
        logger=logger,
        # The per-epoch heartbeat above reports progress to the log.  A tqdm bar
        # writes to stderr with no trailing newline, so under nohup it redrew
        # 1150 of a run's 1392 console lines and welded its text onto the front
        # of every heartbeat line.  Every other trainer here disables it too.
        enable_progress_bar=False,
        num_sanity_val_steps=0,  # Skip validation sanity checks
    )



# Parameters a search trial or a saved best_params entry may set on the config.
LSTM_TUNABLE_PARAMS = (
    "hidden_size", "num_layers", "dropout", "bidirectional",
    "dense_hidden_size", "dense_dropout", "learning_rate", "batch_size",
    "weight_decay", "sequence_length", "target_offset", "embedding_dim",
)


def default_lstm_params() -> Dict:
    """The parameter dict equivalent to LSTMTrainerConfig's own defaults."""
    defaults = LSTMTrainerConfig()
    return {name: getattr(defaults, name) for name in LSTM_TUNABLE_PARAMS}


def lstm_config_from_params(params: Dict, **overrides) -> LSTMTrainerConfig:
    """Build a trainer config from a sampled or persisted parameter dict.

    Anything *params* leaves out falls back to LSTMTrainerConfig's own default,
    so a search trial and the final fit cannot disagree about what a missing
    parameter means.  *overrides* wins over both — that is how a trial gets its
    shortened epochs and single device.

    Integer-typed settings are coerced from whole floats: best_params comes
    back through a CSV, which turns batch_size=128 into 128.0.
    """
    defaults = LSTMTrainerConfig()
    settings = {name: params.get(name, getattr(defaults, name)) for name in LSTM_TUNABLE_PARAMS}

    for name, value in list(settings.items()):
        default = getattr(defaults, name)
        is_integral = isinstance(default, int) and not isinstance(default, bool)
        if is_integral and isinstance(value, float) and value == int(value):
            settings[name] = int(value)

    settings.update(overrides)
    return LSTMTrainerConfig(**settings)


def resolve_feature_columns(train_data, targets, features):
    """Features to model, falling back to every non-index, non-target column."""
    if features is not None:
        return features

    from configs.data import NON_FEATURE_COLUMNS, INDEX_COLUMNS

    # 'Step' is added by sequence preprocessing, not a real feature.
    excluded = set(NON_FEATURE_COLUMNS) | set(INDEX_COLUMNS) | {"Step"} | set(targets)
    return [col for col in train_data.columns if col not in excluded]


def _best_epoch(trainer) -> int:
    """0-based epoch the early-stopping callback judged best.

    After fit, ``current_epoch`` counts completed epochs, so the last one ran
    at index ``current_epoch - 1``, and ``wait_count`` epochs have passed
    since the best.  Matches what the TFT search records, so ``best_epoch``
    means the same thing in both models' ledgers.
    """
    callback = trainer.early_stopping_callback
    return max(0, int(trainer.current_epoch) - 1 - int(getattr(callback, "wait_count", 0)))


def _run_lstm_trial(
    trial_id: int,
    params: Dict,
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    features: List[str],
    targets: List[str],
    run_id: str,
    categorical_features: List[str],
    num_model_families: int,
    num_regions: int,
    devices=1,
    stage: str = "stage1",
    budget: Optional[Dict] = None,
) -> Dict:
    """Train one hyperparameter configuration and return its validation loss.

    A failed trial is recorded with an infinite loss rather than aborting the
    search — a single configuration can exhaust GPU memory without saying
    anything about the rest.
    """
    budget = dict(budget or {})
    try:
        # The same config the final fit would build from these parameters,
        # under whatever budget this stage grants.
        config = lstm_config_from_params(params, devices=devices, **budget)
        train_dataset, val_dataset, model_features = create_lstm_datasets(
            train_data, val_data, features, targets,
            sequence_length=config.sequence_length,
            target_offset=config.target_offset,
            categorical_features=categorical_features,
        )
        train_loader, val_loader = create_lstm_dataloaders(
            train_dataset, val_dataset, batch_size=config.batch_size
        )

        model = create_lstm_model(
            model_features, len(targets), config,
            num_model_families=num_model_families, num_regions=num_regions,
        )

        trial_dir = os.path.join(get_run_root(run_id), "search", f"{stage}_trial_{trial_id}")
        trainer = create_lstm_search_trainer(config, log_dir=os.path.join(trial_dir, "logs"))
        trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        # Rank by the best epoch, as the TFT search does: under early stopping
        # the last epoch is worse than the best by a trial-dependent amount.
        val_loss = trainer.early_stopping_callback.best_score.item()
        logging.info("[%s] Trial %d val_loss: %.4f", stage, trial_id + 1, val_loss)
        return {
            **params,
            "val_loss": val_loss,
            "best_epoch": _best_epoch(trainer),
            "trial_id": trial_id,
            "stage": stage,
            "status": "completed",
        }

    except Exception as e:  # noqa: BLE001 - one trial must not end the search
        logging.error("[%s] Trial %d failed: %s", stage, trial_id + 1, e, exc_info=True)
        return {
            **params,
            "val_loss": float("inf"),
            "trial_id": trial_id,
            "stage": stage,
            "status": "failed",
            "error": str(e),
        }


_TRIAL_BOOKKEEPING_KEYS = ("val_loss", "trial_id", "error", "stage", "status", "best_epoch")


def _report_search_results(
    search_results: List[Dict], run_id: str, space: Optional[LSTMSearchSpace] = None
) -> Dict:
    """Persist the trial table, log the per-sequence-length winners, return the best params."""
    valid_results = [r for r in search_results if np.isfinite(r["val_loss"])]
    if not valid_results:
        raise RuntimeError("All LSTM hyperparameter trials failed")

    # Stage 2 measured its candidates under the full schedule, so it decides
    # whenever it ran; stage 1 only ranks candidates for it.
    stage2 = [r for r in valid_results if r.get("stage") == "stage2"]
    best_result = min(stage2 or valid_results, key=lambda r: r["val_loss"])
    best_score = best_result["val_loss"]
    best_params = {
        k: v for k, v in best_result.items() if k not in _TRIAL_BOOKKEEPING_KEYS
    }

    if space is not None:
        try:
            write_search_report(
                os.path.join(get_run_root(run_id), "search"), space, valid_results
            )
        except Exception as exc:  # noqa: BLE001 - a report must not sink a search
            logging.warning("Failed to write LSTM search report: %s", exc)

    search_results_df = pd.DataFrame(search_results)
    def _format_param(value):
        """Whole numbers should read as whole numbers.

        Collecting the trials into a DataFrame widens the integer
        hyperparameters to float64, so the summary said sequence_length=1.0
        and hidden_size=64.0 for values that are only ever integers.
        """
        if isinstance(value, float) and float(value).is_integer():
            return str(int(value))
        return str(value)

    search_results_path = os.path.join(get_run_root(run_id), "search_results.csv")
    os.makedirs(os.path.dirname(search_results_path), exist_ok=True)
    search_results_df.to_csv(search_results_path, index=False)
    logging.info(f"Search results saved to: {search_results_path}")

    # Log best params per sequence_length and save a CSV report.  Only one
    # stage may contribute: stage-2 rows were trained for far longer, so a
    # table mixing them would rank the lengths that happened to reach stage 2
    # above the rest on budget alone.  Stage 2 is preferred when it ran,
    # because the stratified selection gives every length a row there.
    if "sequence_length" in search_results_df.columns:
        finite_df = search_results_df[np.isfinite(search_results_df["val_loss"])].copy()
        if "stage" in finite_df.columns and (finite_df["stage"] == "stage2").any():
            finite_df = finite_df[finite_df["stage"] == "stage2"].copy()
        if not finite_df.empty:
            idx = finite_df.groupby("sequence_length")["val_loss"].idxmin()
            best_by_seq = finite_df.loc[idx].sort_values("sequence_length")
            best_by_seq_path = os.path.join(get_run_root(run_id), "search_best_by_seq_len.csv")
            best_by_seq.to_csv(best_by_seq_path, index=False)
            logging.info("Best params per sequence_length:")
            for _, row in best_by_seq.iterrows():
                seq = int(row["sequence_length"]) if not pd.isna(row["sequence_length"]) else None
                score = float(row["val_loss"]) if not pd.isna(row["val_loss"]) else None
                params_str = ", ".join(
                    f"{k}={_format_param(row[k])}" for k in search_results_df.columns
                    if k not in _TRIAL_BOOKKEEPING_KEYS and k in row and not pd.isna(row[k])
                )
                logging.info(f"  seq_len={seq}: val_loss={score:.4f} | {params_str}")

    logging.info(f"Best LSTM Params: {best_params} with Val Loss: {best_score:.4f}")
    return best_params


@dataclass
class _TrialInputs:
    """Everything a trial needs beyond its own hyperparameters.

    Bundled into one picklable object because the parallel search ships it to
    a spawned worker per GPU, and threading a dozen positional arguments
    through that hand-off is how they drift apart.
    """

    train_data: pd.DataFrame
    val_data: pd.DataFrame
    features: List[str]
    targets: List[str]
    run_id: str
    categorical_features: List[str]
    num_model_families: int
    num_regions: int


def _search_worker(
    gpu_id: int,
    assignments: List,
    shared_results,
    inputs: "_TrialInputs",
    stage: str,
    budget: Dict,
) -> None:
    """Run one GPU's share of the trials.

    Defined at module level, and taking everything by argument, so it can be
    pickled to a spawned process; a closure would confine the search to the
    fork start method.
    """
    torch.cuda.set_device(gpu_id)
    for trial_id, params in assignments:
        logging.info("[%s] GPU %d - Trial %d: %s", stage, gpu_id, trial_id + 1, params)
        shared_results.append(_run_lstm_trial(
            trial_id, params, inputs.train_data, inputs.val_data, inputs.features,
            inputs.targets, inputs.run_id, inputs.categorical_features,
            inputs.num_model_families, inputs.num_regions,
            devices=[gpu_id], stage=stage, budget=budget,
        ))


def _run_lstm_trials_parallel(
    params_list: List[Dict],
    inputs: "_TrialInputs",
    stage: str,
    budget: Dict,
) -> List[Dict]:
    """Run the trials with one worker process per GPU."""
    import torch.multiprocessing as mp

    num_gpus = torch.cuda.device_count()
    logging.info("[%s] Using %d GPUs for %d trials", stage, num_gpus, len(params_list))

    param_groups = [[] for _ in range(num_gpus)]
    for i, params in enumerate(params_list):
        param_groups[i % num_gpus].append((i, params))

    # Spawn, not the platform default: the CUDA runtime cannot be re-initialised
    # in a forked child, so a fork works only while nothing has touched CUDA in
    # this process before the search — a precondition nothing enforces, and one
    # that fails every trial at once when it breaks.  Spawn costs a fresh
    # interpreter and a pickle of the training frames per worker, which is
    # nothing against a trial's runtime.
    ctx = mp.get_context("spawn")

    with ctx.Manager() as manager:
        shared_results = manager.list()
        processes = []

        for gpu_id, assignments in enumerate(param_groups):
            if not assignments:  # Only start a process if it has work
                continue
            process = ctx.Process(
                target=_search_worker,
                args=(gpu_id, assignments, shared_results, inputs, stage, budget),
            )
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        # A worker killed outright (OOM, segfault) takes its trials with it and
        # would otherwise just look like a shorter search.
        # TODO: it is also only noticed once every worker has finished; see
        # the note on tft_trainer._collect_search_results.
        crashed = [p for p in processes if p.exitcode not in (0, None)]
        if crashed:
            raise RuntimeError(
                "LSTM search worker(s) crashed: "
                + ", ".join(f"pid={p.pid} exitcode={p.exitcode}" for p in crashed)
            )

        return list(shared_results)


def _run_lstm_trials_sequential(
    params_list: List[Dict],
    inputs: "_TrialInputs",
    stage: str,
    budget: Dict,
) -> List[Dict]:
    """Run the trials one at a time."""
    results = []
    for i, params in enumerate(params_list):
        logging.info("[%s] Trial %d/%d - Params: %s", stage, i + 1, len(params_list), params)
        results.append(_run_lstm_trial(
            i, params, inputs.train_data, inputs.val_data, inputs.features,
            inputs.targets, inputs.run_id, inputs.categorical_features,
            inputs.num_model_families, inputs.num_regions,
            stage=stage, budget=budget,
        ))
    return results


def _run_lstm_trials(
    params_list: List[Dict],
    inputs: "_TrialInputs",
    stage: str,
    budget: Dict,
) -> List[Dict]:
    """Run one stage of trials, fanning out over GPUs when there are several."""
    if not params_list:
        return []
    runner = (
        _run_lstm_trials_parallel if torch.cuda.device_count() > 1
        else _run_lstm_trials_sequential
    )
    logging.info("[%s] Using %s LSTM trial execution", stage, runner.__name__.rsplit("_", 1)[-1])
    return runner(params_list, inputs, stage, budget)


def hyperparameter_search_lstm(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    targets: List[str],
    run_id: str,
    features: List[str] = None,
    categorical_features: Optional[List[str]] = None,
    num_model_families: int = 0,
    num_regions: int = 0,
) -> Dict:
    """Two-stage random search, the same protocol the TFT and XGB searches run.

    Stage 1 ranks every sampled configuration under a shortened schedule;
    stage 2 refits the leaders under the full schedule, because a
    configuration that looks good after 20 epochs is not necessarily the one
    that is best after 100.
    """
    space = LSTMSearchSpace()
    inputs = _TrialInputs(
        train_data=train_data,
        val_data=val_data,
        features=resolve_feature_columns(train_data, targets, features),
        targets=targets,
        run_id=run_id,
        categorical_features=categorical_features or [],
        num_model_families=num_model_families,
        num_regions=num_regions,
    )
    logging.info("LSTM search space: %s", space.summary())

    all_params = space.sample()
    stage1_rows = _run_lstm_trials(all_params, inputs, "stage1", space.stage1_budget)

    stage1_done = completed_trials(stage1_rows, space.param_keys)
    if not stage1_done:
        raise RuntimeError("All LSTM stage-1 hyperparameter trials failed")

    signature_to_params = {params_signature(p, space.param_keys): p for p in all_params}
    stage2_params = [
        signature_to_params[sig]
        for sig in select_top_k_signatures(
            stage1_done, space.stage2_top_k, space.param_keys,
            stratify_by=space.stage2_stratify_by,
        )
        if sig in signature_to_params
    ]
    logging.info(
        "LSTM stage2: refitting %d of %d completed stage-1 trials at full budget "
        "(every %s represented)",
        len(stage2_params), len(stage1_done), space.stage2_stratify_by or "leader",
    )
    # Full budget: no overrides, so lstm_config_from_params falls back to
    # LSTMTrainerConfig's max_epochs, with the patience the final fit uses.
    stage2_rows = _run_lstm_trials(
        stage2_params, inputs, "stage2", {"patience": LSTMTrainerConfig().final_patience},
    )

    return _report_search_results(stage1_rows + stage2_rows, run_id, space)


def train_final_lstm(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    targets: List[str],
    run_id: str,
    best_params: Dict,
    session_state: Optional[Dict] = None,
    features: List[str] = None,
    categorical_features: Optional[List[str]] = None,
    num_model_families: int = 0,
    num_regions: int = 0,
) -> None:
    """Train final LSTM model using the same train/val split as the search.

    Uses train_data for training and val_data for early stopping, matching
    the exact data regime the hyperparameter search used to select best_params.
    Per-epoch metrics are logged via CSVLogger.

    Under DDP, only rank 0 logs metrics and saves artifacts to session_state.
    All ranks build the dataset and model (cheap) so trainer.fit() can proceed.
    """

    primary = is_primary_rank()
    categorical_features = categorical_features or []
    features = resolve_feature_columns(train_data, targets, features)

    # Full-length training: no max_epochs/patience override, unlike a trial.
    config = lstm_config_from_params(best_params)

    # Use the same train/val split as the search phase: scalers fit on train_data
    train_dataset, val_dataset, encoded_features = create_lstm_datasets(
        train_data, val_data, features, targets,
        sequence_length=config.sequence_length,
        target_offset=config.target_offset,
        categorical_features=categorical_features,
    )
    non_numeric_cols = _infer_non_numeric_feature_columns(train_data, [f for f in features if f not in categorical_features])

    train_loader, val_loader = create_lstm_dataloaders(
        train_dataset, val_dataset, batch_size=config.batch_size,
    )

    model = create_lstm_model(
        encoded_features, len(targets), config,
        num_model_families=num_model_families,
        num_regions=num_regions,
    )

    # Create final trainer with early stopping and CSV logging
    final_dir = os.path.join(get_run_root(run_id), "final")
    os.makedirs(final_dir, exist_ok=True)

    final_ckpt_path = os.path.join(final_dir, "best.ckpt")
    log_dir = os.path.join(final_dir, "logs")
    trainer = create_lstm_final_trainer(config, ckpt_path=final_ckpt_path, log_dir=log_dir)

    if primary:
        logging.info(
            "LSTM final training: train=%d val=%d, "
            "up to %d epochs with early stopping (patience=%d)",
            len(train_dataset), len(val_dataset),
            config.max_epochs, config.final_patience,
        )

    # Train with validation for early stopping
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if primary:
        train_loss = trainer.callback_metrics.get("train_loss")
        val_loss = trainer.callback_metrics.get("val_loss")
        stopped_epoch = trainer.current_epoch
        if train_loss is not None:
            logging.info(
                "LSTM final training done -> train_loss=%.4f val_loss=%s stopped_epoch=%d",
                train_loss,
                f"{val_loss:.4f}" if val_loss is not None else "NA",
                stopped_epoch,
            )
        else:
            logging.warning("Final training loss not available in callback metrics")

    # ModelCheckpoint already saved the best weights to final_ckpt_path.
    # Save scalers and config to session state (only on primary rank)
    if primary and session_state is not None:
        session_state["lstm_scaler_X"] = train_dataset.scaler_X
        session_state["lstm_scaler_y"] = train_dataset.scaler_y
        session_state["lstm_config"] = config
        session_state["lstm_raw_features"] = features
        session_state["lstm_features"] = encoded_features
        session_state["lstm_non_numeric_features"] = non_numeric_cols
        session_state["lstm_categorical_features"] = categorical_features
        session_state["lstm_num_model_families"] = num_model_families
        session_state["lstm_num_regions"] = num_regions
        session_state["lstm_sequence_length"] = config.sequence_length
        session_state["lstm_target_offset"] = config.target_offset


def predict_lstm(
    session_state: Dict,
    run_id: str,
    *,
    skip_metrics: bool = False,
    metrics_filename: str = "performance.csv",
) -> np.ndarray:
    """Make predictions using trained LSTM model via batched Trainer.predict.

    *skip_metrics* / *metrics_filename* let callers evaluate non-test splits
    (e.g. train/val, for over/underfitting diagnostics) without overwriting
    the canonical test-set "performance.csv".
    """
    from src.trainers.evaluation import save_metrics

    # Get data from session state (stored as DataFrames like TFT)
    test_data = session_state["test_data"]
    targets = session_state["targets"]
    # Prefer encoded LSTM features from final training (ensures scaler/model input size match)
    raw_features = session_state.get("lstm_raw_features", session_state["features"])
    features = session_state.get("lstm_features", session_state["features"])
    categorical_features = session_state.get("lstm_categorical_features", [])

    # Load model
    final_ckpt_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")

    if not os.path.exists(final_ckpt_path):
        raise FileNotFoundError(f"Final model checkpoint not found: {final_ckpt_path}")

    # Get config and scalers from session state
    config = session_state.get("lstm_config")
    scaler_X = session_state.get("lstm_scaler_X")
    scaler_y = session_state.get("lstm_scaler_y")
    sequence_length = session_state.get("lstm_sequence_length", LSTMTrainerConfig().sequence_length)
    target_offset = session_state.get("lstm_target_offset", LSTMTrainerConfig().target_offset)

    if config is None or scaler_X is None or scaler_y is None:
        raise ValueError("LSTM config and scalers not found in session state")

    # Load model
    model = LSTMModel.load_from_checkpoint(final_ckpt_path)
    model.eval()

    # Ensure test_data has the same encoded feature columns as during training
    # Separate continuous features for one-hot encoding
    continuous_raw_features = [f for f in raw_features if f not in categorical_features]
    non_numeric_cols = session_state.get("lstm_non_numeric_features", [])
    if non_numeric_cols and features != continuous_raw_features:
        X_test = test_data.reindex(columns=continuous_raw_features)
        X_test_enc = pd.get_dummies(X_test, columns=[c for c in non_numeric_cols if c in X_test.columns], dummy_na=True)
        X_test_enc = X_test_enc.reindex(columns=features, fill_value=0.0)
        test_data_enc = pd.concat([test_data.drop(columns=continuous_raw_features, errors="ignore"), X_test_enc], axis=1)
    else:
        test_data_enc = test_data

    # Ensure categorical columns are present
    for col in categorical_features:
        if col in test_data.columns and col not in test_data_enc.columns:
            test_data_enc[col] = test_data[col].values

    all_features = features + categorical_features
    test_dataset = LSTMDataset(
        test_data_enc, all_features, targets,
        sequence_length=sequence_length,
        target_offset=target_offset,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        fit_scalers=False,
        categorical_features=categorical_features,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=1,
        persistent_workers=False
    )

    trainer = Trainer(
        devices=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        enable_progress_bar=False,
        logger=False
    )

    raw_predictions = trainer.predict(model, test_loader)
    predictions_list = [p.cpu().numpy() for p in raw_predictions]
    predictions_array = np.vstack(predictions_list)

    if predictions_array.ndim == 3 and predictions_array.shape[1] == 1:
        predictions_array = predictions_array.squeeze(axis=1)

    predictions_unscaled = scaler_y.inverse_transform(predictions_array)

    aligned_preds = align_sequence_predictions(
        test_dataset, predictions_unscaled, len(test_data)
    )

    logging.info("LSTM prediction completed. Shape: %s", aligned_preds.shape)

    # Get valid (non-NaN) predictions and targets for metrics (like TFT pattern)
    y_test = test_data[targets].values

    # Convert per-capita predictions/targets back to absolute units.
    from src.data.preprocess import denormalize_by_population
    from configs.data import POPULATION_COLUMN
    population = test_data[POPULATION_COLUMN].values
    aligned_preds = denormalize_by_population(aligned_preds, population)
    y_test = denormalize_by_population(y_test, population)

    # Pass all rows to save_metrics — it handles NaN per-target internally.
    # Filtering with .any(axis=1) would drop rows where only SOME targets
    # are NaN, biasing per-target R² toward well-covered regions.
    from src.data.preprocess import observed_mask_from_frame
    obs_mask = observed_mask_from_frame(test_data, targets)

    # test_data drives the per-region-scale breakdown; align_sequence_predictions
    # returned one row per test_data row, so the frame matches the scored arrays.
    if not skip_metrics:
        save_metrics(run_id, y_test, aligned_preds, test_data,
                     observed_mask=obs_mask, metrics_filename=metrics_filename)

    # Store horizon data for plotting (like TFT pattern)
    session_state["horizon_df"] = test_data
    session_state["horizon_y_true"] = y_test

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
