from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

from configs.config import RESULTS_PATH

PARAM_DIST = {
        'max_depth': [8, 10, 12],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [300, 500, 700],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 1, 5],
        'reg_alpha': [0.1, 1, 10],
        'reg_lambda': [1, 10, 100],
    }

# for debugging
PARAM_DIST = {
        'max_depth': [5],
        'learning_rate': [0.4],
        'n_estimators': [1000],
        'subsample': [1.0],
        'colsample_bytree': [1.0],
        'gamma': [0],
        'reg_alpha': [5],
        'reg_lambda': [0.1],
    }

PARAM_PAIRS = [
    ('max_depth', 'learning_rate'),
    ('n_estimators', 'subsample'),
    ('colsample_bytree', 'gamma'),
    ('reg_alpha', 'reg_lambda')
]

SEARCH_ITER_N = 30
N_FOLDS = 3


def visualize_multiple_hyperparam_searches(random_search_results, run_id):
    """
    Visualizes the hyperparameter search results using multiple heatmaps for different parameter pairs.
    """
    if random_search_results is None:
        logging.error("No random search results provided.")
        return
    
    results_df = pd.DataFrame(random_search_results)
    results_df['mean_test_score'] = random_search_results['mean_test_score']

    param_search_dir = os.path.join(RESULTS_PATH, run_id, "config")
    os.makedirs(param_search_dir, exist_ok=True)

    for param1, param2 in PARAM_PAIRS:
        heatmap_data = results_df.pivot_table(
            index=f'param_{param1}',
            columns=f'param_{param2}',
            values='mean_test_score',
            aggfunc='mean'
        )
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".1f",
            cmap="coolwarm",
            cbar_kws={'label': 'Mean Test Score'},
            linewidths=.5
        )
        plt.title(f"Hyperparameter Search: {param1} vs {param2}")
        plt.xlabel(param2)
        plt.ylabel(param1)
        plt.tight_layout()

        filename = f"hyperparam_search_{param1}_vs_{param2}.png"
        plt.savefig(os.path.join(param_search_dir, filename))
        plt.close()

def hyperparameter_search(X_train, y_train, run_id):
    # Create the XGBRegressor instance
    xgb = XGBRegressor(
        objective='reg:squarederror',
        n_jobs=-1,
        random_state=0,
        tree_method='hist',
        device='cpu',
    )

    # Set up RandomizedSearchCV
    random_search = RandomizedSearchCV(
        estimator=xgb,
        param_distributions=PARAM_DIST,
        scoring='neg_mean_squared_error',
        cv=N_FOLDS,
        verbose=2,
        n_jobs=-1,
        n_iter=SEARCH_ITER_N,  # Number of parameter settings sampled
        random_state=0
    )

    logging.info("Starting hyperparameter search...")
    random_search.fit(X_train, y_train)
    logging.info("Hyperparameter search completed.")

    # Output the best parameters and score
    logging.info(f"Best parameters: {random_search.best_params_}")
    logging.info(f"Best score: {random_search.best_score_}")
    logging.info(random_search.cv_results_)
    
    # Save the best model in JSON format
    model_dir = os.path.join(RESULTS_PATH, run_id, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "xgb_model.json")
    random_search.best_estimator_.save_model(model_path)
    logging.info(f"Model saved to: {model_path}")

    return random_search.cv_results_

# # for debugging
# def hyperparameter_search(X_train, y_train):
#     # Create the XGBRegressor instance
#     xgb = XGBRegressor(
#         objective='reg:squarederror',
#         n_jobs=-1,
#         random_state=0,
#         max_depth=10,
#         learning_rate=0.2,
#         n_estimators=500,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         gamma=1,
#         reg_alpha=1,
#         reg_lambda=100,
#         scale_pos_weight=100,
#         enable_categorical=True
#     )
#     logging.info("Training XGBRegressor with fixed parameters...")
#     xgb.fit(
#         X_train, 
#         y_train, 
#         eval_set=[(X_train, y_train)], 
#         verbose=True
#     )
#     logging.info("Training completed.")

#     return xgb, None