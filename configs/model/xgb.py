
# Stage 1: Tree Structure
STAGE_1_PARAMS = {
    'max_depth': [5, 9, 13, 17],
    'min_child_weight': [10, 12, 15, 20],
    'gamma': [0],  # Keep gamma at 0 initially
    'eta': [0.4],  # Fixed learning rate
    'num_boost_round': [1000],  # Fixed with early stopping
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 2: Learning Rate and Number of Trees
STAGE_2_PARAMS = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0],  # Keep at 0
    'eta': [0.1, 0.3, 0.4, 0.5],
    'num_boost_round': [300, 500, 700, 1000],
    'reg_alpha': [0],  # Fixed initially
    'reg_lambda': [1],  # Fixed initially
}

# Stage 3: Regularization
STAGE_3_PARAMS = {
    'max_depth': None,  # Will be set from stage 1 best
    'min_child_weight': None,  # Will be set from stage 1 best
    'gamma': [0, 0.1],
    'eta': None,  # Will be set from stage 3 best
    'num_boost_round': None,  # Will be set from stage 3 best
    'reg_alpha': [0, 1, 5, 10],
    'reg_lambda': [0.1, 1, 10]
}
