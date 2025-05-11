from sklearn.model_selection import ParameterSampler
from sklearn.metrics import mean_squared_error
import os
import logging
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
from typing import List, Tuple, Dict
import time
from dask.distributed import Client
import dask.dataframe as dd
import gc
import dask
from dask import delayed
from xgboost.dask import DaskDMatrix, train as dask_train, predict as dask_predict
from configs.config import RESULTS_PATH

# Stage 1: Tree Structure
STAGE_1_PARAMS = {
    'max_depth': [5, 7, 9, 11, 13],
    'min_child_weight': [10, 12, 15, 20],
    'gamma': [0],  # Keep gamma at 0 initially
    'eta': [0.1],  # Fixed learning rate
    'num_boost_round': [1000],  # Fixed with early stopping
    'subsample': [1.0],  # Fixed initially
    'colsample_bytree': [1.0],  # Fixed initially
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 2: Stochastic Parameters
STAGE_2_PARAMS = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0],  # Keep at 0
    'eta': [0.1],  # Fixed learning rate
    'num_boost_round': [1000],  # Fixed with early stopping
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 3: Learning Rate and Number of Trees
STAGE_3_PARAMS = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0],  # Keep at 0
    'eta': [0.01, 0.1, 0.2, 0.3],
    'num_boost_round': [300, 500],
    'subsample': None,  # Will be set from stage 2 best
    'colsample_bytree': None,  # Will be set from stage 2 best
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 4: Regularization
STAGE_4_PARAMS = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0, 0.1],
    'eta': None,  # Will be set from stage 3 best
    'num_boost_round': None,  # Will be set from stage 3 best
    'subsample': None,  # Will be set from stage 2 best
    'colsample_bytree': None,  # Will be set from stage 2 best
    'reg_alpha': [0, 1, 5, 10],
    'reg_lambda': [0.1, 1, 10]
}

SEARCH_ITER_N_PER_STAGE = 15  # Iterations per stage
N_FOLDS = 5

dask.config.set({
    'distributed.worker.memory.target': 0.7,  # Target 70% memory
    'distributed.worker.memory.spill': 0.8,   # Spill at 80%
    'distributed.worker.memory.pause': 0.85,  # Pause at 85%
    'distributed.worker.memory.terminate': 0.95,  # Terminate at 95%
    'distributed.worker.memory.recent-to-old-time': '30s',
    'distributed.logging.distributed': 'warning',
})


def train_and_evaluate_single_config(X_train, y_train, X_val, y_val, targets, params, gpu_id, client, model_path=None) -> Tuple[Dict, float]:
    """
    Train and evaluate a single parameter configuration.
    
    Returns:
    --------
    Tuple[Dict, float]
        A tuple containing the parameters and the negative RMSE score.
    """
    # Select GPU for this iteration
    old_cuda = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        logging.info(f"Training with params: {params}")
        
        # Prepare data
        y_train_df = pd.DataFrame(y_train, columns=targets)
        y_val_df = pd.DataFrame(y_val, columns=targets)
        
        # Scatter data with more conservative settings
        scatter_kwargs = {'hash': False, 'broadcast': True}
        
        # Scatter data in chunks if necessary
        try:
            X_train_future = client.scatter(X_train, **scatter_kwargs)
            y_train_future = client.scatter(y_train_df, **scatter_kwargs)
            X_val_future = client.scatter(X_val, **scatter_kwargs)
            y_val_future = client.scatter(y_val_df, **scatter_kwargs)
        except Exception as scatter_error:
            logging.warning(f"Direct scatter failed, trying chunked approach: {scatter_error}")
            # If scatter fails due to memory, try chunking
            chunk_size = len(X_train) // 4
            X_train_chunks = [X_train.iloc[i:i+chunk_size] for i in range(0, len(X_train), chunk_size)]
            y_train_chunks = [y_train_df.iloc[i:i+chunk_size] for i in range(0, len(y_train), chunk_size)]
            X_val_chunks = [X_val.iloc[i:i+chunk_size] for i in range(0, len(X_val), chunk_size)]
            y_val_chunks = [y_val_df.iloc[i:i+chunk_size] for i in range(0, len(y_val), chunk_size)]
            
            X_train_futures = [client.scatter(chunk, **scatter_kwargs) for chunk in X_train_chunks]
            y_train_futures = [client.scatter(chunk, **scatter_kwargs) for chunk in y_train_chunks]
            X_val_futures = [client.scatter(chunk, **scatter_kwargs) for chunk in X_val_chunks]
            y_val_futures = [client.scatter(chunk, **scatter_kwargs) for chunk in y_val_chunks]
            
            # Combine chunks on workers
            X_train_future = client.submit(pd.concat, X_train_futures, **scatter_kwargs)
            y_train_future = client.submit(pd.concat, y_train_futures, **scatter_kwargs)
            X_val_future = client.submit(pd.concat, X_val_futures, **scatter_kwargs)
            y_val_future = client.submit(pd.concat, y_val_futures, **scatter_kwargs)
        
        # Create DataFrames from futures using delayed
        X_train_dask = dd.from_delayed([delayed(lambda x: x)(X_train_future)], meta=X_train)
        y_train_dask = dd.from_delayed([delayed(lambda x: x)(y_train_future)], meta=y_train_df)
        X_val_dask = dd.from_delayed([delayed(lambda x: x)(X_val_future)], meta=X_val)
        y_val_dask = dd.from_delayed([delayed(lambda x: x)(y_val_future)], meta=y_val_df)
        
        # Create DaskDMatrix
        dtrain = DaskDMatrix(client, X_train_dask, y_train_dask)
        dval = DaskDMatrix(client, X_val_dask, y_val_dask)
        
        # Set XGBoost parameters
        xgb_params = {
            'tree_method': 'hist',
            'device': 'cuda',
            'eval_metric': 'rmse',
            'verbosity': 0,
            'max_bin': 256,  # Reduce memory usage
            **params
        }
        
        # Train model
        model = dask_train(
            client,
            xgb_params,
            dtrain,
            evals=[(dval, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        # Make predictions with retry and better error handling
        max_retries = 3
        for attempt in range(max_retries):
            try:
                client.run(gc.collect)  # Force garbage collection
                
                predictions = dask_predict(client, model, dval)
                
                # Compute predictions in worker memory
                if hasattr(predictions, 'compute'):
                    predictions = predictions.compute()
                elif hasattr(predictions, 'result'):
                    predictions = predictions.result()
                
                break
            except Exception as pred_error:
                if attempt == max_retries - 1:
                    raise pred_error
                logging.warning(f"Prediction attempt {attempt + 1} failed, retrying...")
                
                # Force cleanup and retry
                try:
                    if 'dval' in locals():
                        del dval
                        dval = DaskDMatrix(client, X_val_dask, y_val_dask)
                    client.run(gc.collect)
                    time.sleep(2)
                except:
                    pass
        
        # Calculate negative RMSE (higher is better)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        score = -rmse  # Negative RMSE for maximization
        
        if model_path:
            model['booster'].save_model(model_path)
            logging.info(f"Model saved to {model_path}")
        
        # Explicit cleanup
        try:
            del dtrain, dval, model, predictions
            del X_train_future, y_train_future, X_val_future, y_val_future
            client.cancel([X_train_future, y_train_future, X_val_future, y_val_future])
        except:
            pass
            
        return params, score
        
    except Exception as e:
        logging.error(f"Error training config {params}: {str(e)}")
        raise
    finally:
        # Restore original CUDA setting
        if old_cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)


def hyperparameter_search(
    X_train: pd.DataFrame, 
    y_train: np.array, 
    X_val: pd.DataFrame, 
    y_val: np.array, 
    targets: List[str], 
    run_id: str, 
    start_stage: int = 1
) -> Tuple[Dict, Dict]:
    """
    Perform staged hyperparameter search for XGBoost.
    
    This function performs hyperparameter search in 4 stages:
    1. Tree structure (max_depth, min_child_weight)
    2. Stochastic parameters (subsample, colsample_bytree)
    3. Learning rate and number of trees (eta, num_boost_round)
    4. Regularization (gamma, reg_alpha, reg_lambda)
    
    Returns:
    --------
    Tuple[Dict, Dict]
        A tuple containing:
        - best_params (Dict): The final best hyperparameter combination
        - all_results (Dict): Results from all stages for visualization
    """
    # Create checkpoint directory
    checkpoint_dir = os.path.join(RESULTS_PATH, run_id, "checkpoints", "staged_search")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Store original CUDA_VISIBLE_DEVICES
    original_cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    # Initialize results tracking
    all_results = {
        'stage_1': [],
        'stage_2': [],
        'stage_3': [],
        'stage_4': []
    }
    
    best_params = {}
    overall_best_score = float('-inf')
    overall_best_params = None
    os.makedirs(os.path.join(RESULTS_PATH, run_id, "checkpoints"), exist_ok=True)
    
    # Define the stages
    stages = [
        ("Stage 1: Tree Structure", STAGE_1_PARAMS),
        ("Stage 2: Stochastic Parameters", STAGE_2_PARAMS),
        ("Stage 3: Learning Rate & Trees", STAGE_3_PARAMS),
        ("Stage 4: Regularization", STAGE_4_PARAMS)
    ]
    # Load best parameters from previous stages if resuming
    for stage_idx in range(start_stage - 1):
        checkpoint_file = os.path.join(checkpoint_dir, f"stage_{stage_idx + 1}_best.json")
        if os.path.exists(checkpoint_file):
            logging.info(f"Loading best parameters from {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                stage_data = json.load(f)
                best_params.update(stage_data['params'])
        else:
            logging.warning(f"Checkpoint for Stage {stage_idx + 1} not found. Starting fresh.")
    
    try:
        for stage_idx, (stage_name, stage_params) in enumerate(stages):
            if stage_idx + 1 < start_stage:
                continue
            
            logging.info(f"{'='*50}")
            logging.info(f"Starting {stage_name}")
            logging.info(f"{'='*50}")
            
            # Prepare parameter distribution for current stage
            current_param_dist = {}
            for param, values in stage_params.items():
                if values is None:
                    # Use best value from previous stage
                    if param in best_params:
                        current_param_dist[param] = [best_params[param]]
                    else:
                        logging.warning(f"Parameter {param} not found in previous stages, skipping...")
                else:
                    current_param_dist[param] = values
            
            # Skip if no parameters to search
            if not current_param_dist:
                logging.warning(f"No parameters to search in {stage_name}, skipping...")
                continue
            
            # Initialize tracking for this stage
            stage_results = []
            stage_best_score = float('-inf')
            stage_best_params = None
            
            # Sample parameters for this stage
            param_sampler = ParameterSampler(
                current_param_dist, 
                n_iter=SEARCH_ITER_N_PER_STAGE, 
                random_state=stage_idx
            )
            
            for i, params in enumerate(param_sampler):
                logging.info(f"{stage_name} - Iteration {i+1}/{SEARCH_ITER_N_PER_STAGE}")
                
                # Select GPU for this iteration
                gpu_id = i % 8
                
                client = None
                try:
                    # Create client with optimized memory settings
                    client = Client(
                        n_workers=1,
                        threads_per_worker=2,
                        memory_limit='4GB',
                        silence_logs=logging.WARNING,
                        dashboard_address=None,
                        local_directory='/tmp/dask-worker-space'
                    )
                    
                    # Train and evaluate
                    params_copy, score = train_and_evaluate_single_config(
                        X_train, y_train, X_val, y_val, targets, params, gpu_id, client
                    )
                    
                    # Store results
                    result = params_copy.copy()
                    result['mean_test_score'] = score
                    result['stage'] = stage_idx + 1
                    stage_results.append(result)
                    
                    # Track stage best
                    if score > stage_best_score:
                        stage_best_score = score
                        stage_best_params = params_copy.copy()
                    
                    # Track overall best
                    if score > overall_best_score:
                        overall_best_score = score
                        overall_best_params = params_copy.copy()
                        
                        model_path = os.path.join(RESULTS_PATH, run_id, "checkpoints", f"final_best.json")
                        train_and_evaluate_single_config(
                            X_train, y_train, X_val, y_val, targets, params_copy, gpu_id, client, model_path
                        )
                        
                    logging.info(f"RMSE: {-score:.4f}")
                    
                except Exception as e:
                    logging.error(f"Error in {stage_name} iteration {i+1}: {str(e)}")
                    logging.error(f"Full error details:", exc_info=True)
                    continue
                    
                finally:
                    # Ensure client is properly closed
                    if client is not None:
                        try:
                            client.run(gc.collect)
                            client.close()
                        except:
                            pass
                    
                    # Force local garbage collection
                    gc.collect()
                    
                    # Small delay to allow cleanup
                    time.sleep(0.5)
            
            # Update best_params with stage results
            if stage_best_params:
                for param in current_param_dist.keys():
                    if param in stage_best_params:
                        best_params[param] = stage_best_params[param]
                
                # Save stage checkpoint
                checkpoint_file = os.path.join(checkpoint_dir, f"stage_{stage_idx + 1}_best.json")
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'stage': stage_idx + 1,
                        'params': stage_best_params,
                        'score': stage_best_score,
                        'rmse': -stage_best_score
                    }, f, indent=2)
            
            # Store stage results
            stage_key = f'stage_{stage_idx + 1}'
            all_results[stage_key] = stage_results
            
            logging.info(f"\n{stage_name} Complete")
            logging.info(f"Stage Best RMSE: {-stage_best_score:.4f}")
            logging.info(f"Stage Best Params: {stage_best_params}")
            
        # Restore original CUDA_VISIBLE_DEVICES
        if original_cuda_devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = original_cuda_devices
        else:
            os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        
        # Final summary
        logging.info(f"{'='*50}")
        logging.info("STAGED SEARCH COMPLETE")
        logging.info(f"{'='*50}")
        logging.info(f"Overall Best RMSE: {-overall_best_score:.4f}")
        logging.info(f"Final Best Parameters: {overall_best_params}")
        
        # Save final results
        final_checkpoint = os.path.join(checkpoint_dir, "final_best.json")
        
        return overall_best_params, all_results
        
    except Exception as e:
        logging.error(f"Error in staged hyperparameter search: {str(e)}")
        raise


def visualize_staged_search_results(all_results: Dict, run_id: str):
    """
    Visualize the results from staged hyperparameter search.
    
    Creates visualization for each stage showing parameter exploration and performance.
    """
    if not all_results:
        logging.error("No staged search results provided.")
        return
    
    vis_dir = os.path.join(RESULTS_PATH, run_id, "config", "staged_search")
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create overall summary plot
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Performance across stages
    plt.subplot(2, 2, 1)
    stage_best_scores = []
    stage_labels = []
    
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            scores = [r['mean_test_score'] for r in all_results[stage_key]]
            if scores:
                stage_best_scores.append(max(scores))
                stage_labels.append(f'Stage {stage_num}')
    
    if stage_best_scores:
        plt.bar(stage_labels, [-s for s in stage_best_scores])  # Convert back to positive RMSE
        plt.ylabel('Best RMSE')
        plt.title('Best Performance by Stage')
        plt.xticks(rotation=45)
    
    # Plot 2: Parameter evolution
    plt.subplot(2, 2, 2)
    param_evolution = {}
    
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            for result in all_results[stage_key]:
                if result['mean_test_score'] == max(r['mean_test_score'] for r in all_results[stage_key]):
                    for param, value in result.items():
                        if param not in ['mean_test_score', 'stage']:
                            if param not in param_evolution:
                                param_evolution[param] = []
                            param_evolution[param].append((stage_num, value))
    
    for param, values in param_evolution.items():
        stages, vals = zip(*values)
        plt.plot(stages, vals, marker='o', label=param)
    
    plt.xlabel('Stage')
    plt.ylabel('Parameter Value')
    plt.title('Parameter Evolution Across Stages')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Stage progression
    plt.subplot(2, 2, 3)
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            scores = [-r['mean_test_score'] for r in all_results[stage_key]]  # Convert to positive RMSE
            plt.plot(scores, alpha=0.7, label=f'Stage {stage_num}')
    
    plt.xlabel('Iteration within Stage')
    plt.ylabel('RMSE')
    plt.title('Search Progress by Stage')
    plt.legend()
    
    # Plot 4: Final parameters heatmap (if possible)
    plt.subplot(2, 2, 4)
    final_stage_results = all_results.get('stage_4', [])
    if final_stage_results and len(final_stage_results) > 1:
        # Create a simple heatmap of final stage parameters
        df = pd.DataFrame(final_stage_results)
        numeric_cols = df.select_dtypes(include=[np.number]).drop('stage', axis=1, errors='ignore')
        
        if not numeric_cols.empty:
            corr_matrix = numeric_cols.corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
            plt.title('Final Stage Parameter Correlations')
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'staged_search_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create individual stage visualizations
    for stage_num in range(1, 5):
        stage_key = f'stage_{stage_num}'
        if stage_key in all_results and all_results[stage_key]:
            visualize_single_stage(all_results[stage_key], stage_num, vis_dir)


def visualize_single_stage(stage_results: List[Dict], stage_num: int, vis_dir: str):
    """
    Create visualization for a single stage of hyperparameter search.
    """
    df = pd.DataFrame(stage_results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Stage {stage_num} Hyperparameter Search Results')
    
    # Plot 1: Parameter distribution
    ax1 = axes[0, 0]
    param_cols = [col for col in df.columns if col.startswith('param_') or (col not in ['mean_test_score', 'stage'])]
    if param_cols:
        df[param_cols].boxplot(ax=ax1)
        ax1.set_title('Parameter Value Distributions')
        ax1.tick_params(axis='x', rotation=90)
    
    # Plot 2: Score distribution
    ax2 = axes[0, 1]
    ax2.hist(-df['mean_test_score'], bins=20, edgecolor='black')  # Convert to positive RMSE
    ax2.set_xlabel('RMSE')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Score Distribution')
    
    # Plot 3: Top 5 configurations
    ax3 = axes[1, 0]
    top_5 = df.nlargest(5, 'mean_test_score')
    top_5_data = []
    
    for _, row in top_5.iterrows():
        config_str = ', '.join([f"{k}: {v}" for k, v in row.items() if k not in ['mean_test_score', 'stage']])
        top_5_data.append((config_str, -row['mean_test_score']))  # Convert to positive RMSE
    
    if top_5_data:
        configs, scores = zip(*top_5_data)
        y_pos = np.arange(len(configs))
        ax3.barh(y_pos, scores)
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([c[:50] + '...' if len(c) > 50 else c for c in configs])
        ax3.set_xlabel('RMSE')
        ax3.set_title('Top 5 Configurations')
    
    # Plot 4: Parameter impact (if enough data)
    ax4 = axes[1, 1]
    if len(df) > 10:
        # Calculate parameter importance
        param_importance = {}
        for col in param_cols:
            if df[col].dtype in [np.float64, np.int64] and df[col].nunique() > 1:
                corr = df[[col, 'mean_test_score']].corr().iloc[0, 1]
                param_importance[col] = abs(corr)
        
        if param_importance:
            params, importance = zip(*sorted(param_importance.items(), key=lambda x: x[1], reverse=True))
            ax4.bar(params, importance)
            ax4.set_ylabel('Absolute Correlation with Score')
            ax4.set_title('Parameter Importance')
            ax4.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, f'stage_{stage_num}_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()