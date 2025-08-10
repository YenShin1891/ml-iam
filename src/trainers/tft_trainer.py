import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List
from sklearn.model_selection import ParameterSampler

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, RMSE

from configs.config import RESULTS_PATH
from configs.models import TFTSearchSpace, TFTTrainerConfig


search_cfg = TFTSearchSpace()
trainer_cfg = TFTTrainerConfig()

PARAM_DIST = search_cfg.param_dist
SEARCH_ITER_N = search_cfg.search_iter_n


def hyperparameter_search_tft(
    train_dataset, 
    val_dataset, 
    targets: List[str],
    run_id: str,
) -> Tuple[object, dict, dict]:

    search_results = []
    best_score = float("inf")
    best_model = None
    best_params = None

    for i, params in enumerate(ParameterSampler(PARAM_DIST, n_iter=SEARCH_ITER_N, random_state=0)):
        logging.info(f"TFT Search Iteration {i+1}/{SEARCH_ITER_N} - Params: {params}")

        output_size = len(targets)

        tft = TemporalFusionTransformer.from_dataset(
            train_dataset,
            hidden_size=params["hidden_size"],
            lstm_layers=params["lstm_layers"],
            dropout=params["dropout"],
            learning_rate=params["learning_rate"],
            output_size=output_size,
            loss=RMSE(),
            log_interval=0,
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            dirpath=os.path.join(RESULTS_PATH, run_id, "checkpoints"),
            filename="best_model_{epoch:02d}-{val_loss:.4f}"
        )
        trainer = Trainer(
            max_epochs=trainer_cfg.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=trainer_cfg.devices,
            gradient_clip_val=trainer_cfg.gradient_clip_val,
            callbacks=[EarlyStopping(monitor="val_loss", patience=trainer_cfg.patience), checkpoint_callback],
            logger=False,
            enable_checkpointing=True,
        )

        train_dataloader = train_dataset.to_dataloader(train=True, batch_size=trainer_cfg.batch_size)
        val_dataloader = val_dataset.to_dataloader(train=False, batch_size=trainer_cfg.batch_size)

        trainer.fit(model=tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        val_loss = trainer.callback_metrics["val_loss"].item()
        search_results.append({**params, "val_loss": val_loss})

        if val_loss < best_score:
            best_score = val_loss
            best_model = tft
            best_params = params

    cv_results_dict = {key: [r.get(key) for r in search_results] for key in PARAM_DIST.keys()}
    cv_results_dict["val_loss"] = [r["val_loss"] for r in search_results]

    logging.info(f"Best TFT Params: {best_params} with Val Loss: {best_score:.4f}")

    return best_model, best_params, cv_results_dict

def visualize_multiple_hyperparam_searches_tft(cv_results_dict, run_id):
    results_df = pd.DataFrame(cv_results_dict)
    param_search_dir = os.path.join(RESULTS_PATH, run_id, "config")
    os.makedirs(param_search_dir, exist_ok=True)

    # derive parameter keys automatically (exclude target metric)
    keys = [k for k in results_df.columns if k != "val_loss"]

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            param1, param2 = keys[i], keys[j]
            try:
                heatmap_data = results_df.pivot_table(
                    index=param1,
                    columns=param2,
                    values="val_loss",
                    aggfunc="mean",
                )
            except Exception:
                continue

            plt.figure(figsize=(10, 8))
            sns.heatmap(
                heatmap_data,
                annot=True,
                fmt=".4f",
                cmap="coolwarm",
                cbar_kws={"label": "Validation Loss"},
                linewidths=0.5,
            )
            plt.title(f"TFT Hyperparam Search: {param1} vs {param2}")
            plt.xlabel(param2)
            plt.ylabel(param1)
            plt.tight_layout()

            filename = f"tft_hyperparam_search_{param1}_vs_{param2}.png"
            plt.savefig(os.path.join(param_search_dir, filename))
            plt.close()
