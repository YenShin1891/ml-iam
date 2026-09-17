#!/usr/bin/env python

"""Learned categorical embeddings of a trained LSTM/TFT run.

Both sequence models embed exactly two categorical inputs, Region and
Model_Family (configs/data.py:CATEGORICAL_COLUMNS). This script reads the
embedding tables out of final/best.ckpt and asks whether known structure
shows up in them without being told:

  plots/embeddings/model_family_pca.png        PCA (or t-SNE) of the family table
  plots/embeddings/model_family_heatmap.png    cosine similarity, cluster-ordered
  plots/embeddings/model_family_neighbours.csv top-3 nearest families per family
  plots/embeddings/region_pca.png              PCA of the region table, by scale
  plots/embeddings/region_neighbours.csv       rank of each country's macro-region

Rows whose label never appears in the training split are random init and
are dropped (the LSTM vocabulary is the train+val+test union). The train
row count is printed next to every family so the thinly-trained ones are
read with care.

PCA is the default: with 14-16 families, t-SNE draws artefacts more than
structure. It stays available behind --method tsne.

Usage:
  python scripts/plot_embeddings.py --model lstm --run_id lstm_89
  python scripts/plot_embeddings.py --model tft  --run_id tft_94 --method tsne
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs.data import CATEGORICAL_COLUMNS, INDEX_COLUMNS  # noqa: E402
from configs.visualization import LEGEND_FONTSIZE, PLOT_FONT_SIZE, TICK_LABELSIZE  # noqa: E402
from src.utils.regions import SCALE_ORDER_COARSEST_FIRST, region_scale  # noqa: E402
from src.utils.run_store import RunStore  # noqa: E402
from src.utils.utils import get_run_root  # noqa: E402

# Solution concept of each model family, for colouring the family plot.
# Assignments follow the AR6 WGIII Annex III model descriptions; PE = partial
# equilibrium / energy-system models. Families absent here draw grey.
MODEL_FAMILY_TYPE: Dict[str, str] = {
    "AIM": "CGE",
    "GEM": "CGE",
    "IMACLIM": "CGE",
    "C3IAM": "CGE",
    "MESSAGEix": "energy-system PE",
    "TIAM": "energy-system PE",
    "POLES": "energy-system PE",
    "IMAGE": "energy-system PE",
    "GCAM": "energy-system PE",
    "MARKAL": "energy-system PE",
    "LUT": "energy-system PE",
    "PyPSA": "energy-system PE",
    "DDPP": "energy-system PE",
    "REMIND": "intertemporal optimisation",
    "WITCH": "intertemporal optimisation",
    "MERGE": "intertemporal optimisation",
}
MODEL_TYPE_COLOURS = {
    "CGE": "#d95f02",
    "energy-system PE": "#1b9e77",
    "intertemporal optimisation": "#7570b3",
    "unknown": "#999999",
}

# Country -> the R10 macro-region that contains it. If the model has learned
# geography, the macro-region should be among the country's nearest neighbours.
COUNTRY_MACRO_REGION: Dict[str, str] = {
    "CHN": "R10CHINA+",
    "IND": "R10INDIA+",
    "USA": "R10NORTH_AM",
    "CAN": "R10NORTH_AM",
    "EU": "R10EUROPE",
    "JPN": "R10PAC_OECD",
    "KOR": "R10PAC_OECD",
    "AUS": "R10PAC_OECD",
    "BRA": "R10LATIN_AM",
    "MEX": "R10LATIN_AM",
    "RUS": "R10REF_ECON",
    "SAU": "R10MIDDLE_EAST",
    "ZAF": "R10AFRICA",
    "IDN": "R10REST_ASIA",
}

SCALE_COLOURS = {
    "World": "#000000",
    "R5": "#e7298a",
    "R6": "#66a61e",
    "R10": "#1b9e77",
    "ISO3": "#7570b3",
}


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------

def _load_lstm_tables(run_id: str) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Embedding rows of the LSTM, labelled by the run's categories.json."""
    import torch

    ckpt_path = os.path.join(get_run_root(run_id), "final", "best.ckpt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Final LSTM checkpoint not found at {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    categories = RunStore(run_id).load_categories()

    keys = {"Region": "region_embedding.weight", "Model_Family": "model_family_embedding.weight"}
    tables = {}
    for col, key in keys.items():
        if key not in state:
            continue
        weight = state[key].numpy()
        labels = list(categories[col])
        if weight.shape[0] != len(labels):
            raise ValueError(
                f"{key} has {weight.shape[0]} rows but categories.json lists "
                f"{len(labels)} {col} labels; the checkpoint and vocabulary disagree."
            )
        tables[col] = (labels, weight)
    return tables


def _load_tft_tables(run_id: str) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Embedding rows of the TFT, labelled by the run's fitted label encoders."""
    from src.trainers.tft_dataset import load_dataset_template
    from src.trainers.tft_model import load_tft_checkpoint

    model = load_tft_checkpoint(run_id, map_location="cpu")
    encoders = load_dataset_template(run_id).categorical_encoders

    tables = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in model.input_embeddings.embeddings:
            continue
        weight = model.input_embeddings.embeddings[col].weight.detach().cpu().numpy()
        classes = encoders[col].classes_  # {label: row}
        if sorted(classes.values()) != list(range(weight.shape[0])):
            raise ValueError(
                f"{col} encoder rows are not a permutation of range({weight.shape[0]})."
            )
        labels = [None] * weight.shape[0]
        for label, row in classes.items():
            labels[row] = str(label)
        tables[col] = (labels, weight)
    return tables


def _train_row_counts(run_id: str) -> Dict[str, pd.Series]:
    """Training rows per Region / Model_Family label, from the run's own split."""
    store = RunStore(run_id)
    splits = store.load_splits()
    data = store.load_processed_data()[INDEX_COLUMNS + ["Model_Family"]]
    train = data.merge(splits[splits["split"] == "train"], on=INDEX_COLUMNS, how="inner")
    return {col: train.groupby(col).size() for col in CATEGORICAL_COLUMNS}


def _keep_trained(labels: List[str], weight: np.ndarray, counts: pd.Series):
    """Drop rows whose label the training split never contained."""
    n_train = np.array([int(counts.get(label, 0)) for label in labels])
    keep = n_train > 0
    dropped = [label for label, k in zip(labels, keep) if not k]
    return (
        [label for label, k in zip(labels, keep) if k],
        weight[keep],
        n_train[keep],
        dropped,
    )


# ----------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------

def _cosine_similarity(weight: np.ndarray) -> np.ndarray:
    unit = weight / np.linalg.norm(weight, axis=1, keepdims=True)
    return unit @ unit.T


def _project_2d(weight: np.ndarray, method: str, seed: int = 0):
    """2D coordinates and an axis-label pair for the chosen method."""
    if method == "pca":
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=seed)
        coords = pca.fit_transform(weight)
        var = pca.explained_variance_ratio_ * 100
        return coords, (f"PC1 ({var[0]:.0f}% var)", f"PC2 ({var[1]:.0f}% var)")

    from sklearn.manifold import TSNE

    perplexity = max(2, min(30, (weight.shape[0] - 1) // 3))
    coords = TSNE(
        n_components=2, perplexity=perplexity, init="pca", random_state=seed
    ).fit_transform(weight)
    return coords, (f"t-SNE 1 (perplexity {perplexity})", "t-SNE 2")


def _nearest_neighbours(labels: List[str], sim: np.ndarray, k: int) -> pd.DataFrame:
    rows = []
    for i, label in enumerate(labels):
        order = [j for j in np.argsort(-sim[i]) if j != i][:k]
        row = {"label": label}
        for rank, j in enumerate(order, start=1):
            row[f"nn{rank}"] = labels[j]
            row[f"nn{rank}_cosine"] = round(float(sim[i, j]), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _macro_region_ranks(labels: List[str], sim: np.ndarray) -> pd.DataFrame:
    """Where each country's containing R10 region ranks among its neighbours."""
    index = {label: i for i, label in enumerate(labels)}
    rows = []
    for country, macro in COUNTRY_MACRO_REGION.items():
        if country not in index or macro not in index:
            continue
        i = index[country]
        order = [j for j in np.argsort(-sim[i]) if j != i]
        rank = order.index(index[macro]) + 1
        rows.append({
            "country": country,
            "macro_region": macro,
            "rank_of_macro_region": rank,
            "of_n_neighbours": len(order),
            "cosine": round(float(sim[i, index[macro]]), 3),
            "nn1": labels[order[0]],
            "nn2": labels[order[1]],
            "nn3": labels[order[2]],
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------------

def _plot_family_scatter(labels, weight, n_train, method, title, path):
    import matplotlib.pyplot as plt

    coords, axis_labels = _project_2d(weight, method)
    fig, ax = plt.subplots(figsize=(9, 8))
    for label, (x, y), n in zip(labels, coords, n_train):
        kind = MODEL_FAMILY_TYPE.get(label, "unknown")
        ax.scatter(x, y, s=80, color=MODEL_TYPE_COLOURS[kind], edgecolor="black", zorder=3)
        ax.annotate(f"{label}\n(n={n:,})", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=TICK_LABELSIZE - 3)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                   markerfacecolor=colour, markeredgecolor="black", label=kind)
        for kind, colour in MODEL_TYPE_COLOURS.items()
        if kind != "unknown" or any(l not in MODEL_FAMILY_TYPE for l in labels)
    ]
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, loc="best", title="solution concept")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_similarity_heatmap(labels, sim, title, path):
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform

    distance = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    order = leaves_list(linkage(squareform(distance, checks=False), method="average"))
    ordered = [labels[i] for i in order]
    block = sim[np.ix_(order, order)]

    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.55 * n + 3, 0.55 * n + 2))
    image = ax.imshow(block, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ordered, rotation=90, fontsize=TICK_LABELSIZE - 4)
    ax.set_yticklabels(ordered, fontsize=TICK_LABELSIZE - 4)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{block[i, j]:.2f}", ha="center", va="center",
                    fontsize=TICK_LABELSIZE - 8,
                    color="white" if abs(block[i, j]) > 0.6 else "black")
    fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02, label="cosine similarity")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_region_scatter(labels, weight, method, title, path):
    import matplotlib.pyplot as plt

    coords, axis_labels = _project_2d(weight, method)
    scales = [region_scale(label) for label in labels]
    fig, ax = plt.subplots(figsize=(11, 9))
    for scale in SCALE_ORDER_COARSEST_FIRST:
        idx = [i for i, s in enumerate(scales) if s == scale]
        if not idx:
            continue
        ax.scatter(coords[idx, 0], coords[idx, 1], s=60, label=scale,
                   color=SCALE_COLOURS.get(scale, "#999999"), edgecolor="black", zorder=3)
    for label, scale, (x, y) in zip(labels, scales, coords):
        if scale in {"ISO3", "R10", "World"}:
            ax.annotate(label.replace("R10", ""), (x, y), textcoords="offset points",
                        xytext=(4, 3), fontsize=TICK_LABELSIZE - 5)
    ax.legend(fontsize=LEGEND_FONTSIZE, title="region scale")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["lstm", "tft"], help="Model type.")
    parser.add_argument("--run_id", required=True, help="Trained run id, e.g. lstm_89 / tft_94.")
    parser.add_argument("--method", default="pca", choices=["pca", "tsne"], help="2D projection (default: pca).")
    parser.add_argument("--output_dir", default=None, help="Default: <run_root>/plots/embeddings.")
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": PLOT_FONT_SIZE - 2})

    run_root = get_run_root(args.run_id)
    output_dir = args.output_dir or os.path.join(run_root, "plots", "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    tables = _load_lstm_tables(args.run_id) if args.model == "lstm" else _load_tft_tables(args.run_id)
    if not tables:
        raise ValueError(f"{args.run_id} has no categorical embeddings in its checkpoint.")
    counts = _train_row_counts(args.run_id)
    tag = f"{args.model.upper()} {args.run_id}"

    # --- Model family -------------------------------------------------------
    if "Model_Family" in tables:
        labels, weight = tables["Model_Family"]
        labels, weight, n_train, dropped = _keep_trained(labels, weight, counts["Model_Family"])
        print(f"Model_Family: {len(labels)} families x {weight.shape[1]}-dim"
              + (f"; dropped untrained {dropped}" if dropped else ""))
        sim = _cosine_similarity(weight)

        _plot_family_scatter(labels, weight, n_train, args.method,
                             f"{tag}: Model_Family embedding ({args.method.upper()})",
                             os.path.join(output_dir, f"model_family_{args.method}.png"))
        _plot_similarity_heatmap(labels, sim, f"{tag}: Model_Family cosine similarity",
                                 os.path.join(output_dir, "model_family_heatmap.png"))
        neighbours = _nearest_neighbours(labels, sim, k=3)
        neighbours.insert(1, "n_train", n_train)
        neighbours.insert(2, "type", [MODEL_FAMILY_TYPE.get(l, "unknown") for l in labels])
        neighbours.to_csv(os.path.join(output_dir, "model_family_neighbours.csv"), index=False)
        print(neighbours[["label", "n_train", "type", "nn1", "nn1_cosine", "nn2", "nn3"]].to_string(index=False))

    # --- Region --------------------------------------------------------------
    if "Region" in tables:
        labels, weight = tables["Region"]
        labels, weight, n_train, dropped = _keep_trained(labels, weight, counts["Region"])
        print(f"\nRegion: {len(labels)} regions x {weight.shape[1]}-dim"
              + (f"; dropped {len(dropped)} untrained: {dropped}" if dropped else ""))
        sim = _cosine_similarity(weight)

        _plot_region_scatter(labels, weight, args.method,
                             f"{tag}: Region embedding ({args.method.upper()})",
                             os.path.join(output_dir, f"region_{args.method}.png"))
        ranks = _macro_region_ranks(labels, sim)
        ranks.to_csv(os.path.join(output_dir, "region_neighbours.csv"), index=False)
        if not ranks.empty:
            print(ranks.to_string(index=False))
            median = int(ranks["rank_of_macro_region"].median())
            print(f"-> median rank of a country's own R10 region: {median} of "
                  f"{int(ranks['of_n_neighbours'].iloc[0])} (chance ~ "
                  f"{int(ranks['of_n_neighbours'].iloc[0]) // 2})")

    print(f"\nSaved embedding plots to {output_dir}")


if __name__ == "__main__":
    main()
