from Data import processor
from Models import baseline_models, baseline_strategies, baseline_model_trainer
import matplotlib.pyplot as plt


###############################################
# This module initializes and trains baseline models #
###############################################

def initialize(stock, start, end, test_ratio):
    _, X_train, _, _, _, _, _ = processor.baseline_preprocessing(stock, start, end, test_ratio)
    LASSO = baseline_models.Linear_Model(input_shape=X_train.shape[1], output_shape=1, model_type='lasso', alpha=0.2)
    RIDGE = baseline_models.Linear_Model(input_shape=X_train.shape[1], output_shape=1, model_type='ridge', alpha=0.2)
    LR = baseline_models.Linear_Model(input_shape=X_train.shape[1], output_shape=1)
    RANDOM_FOREST = baseline_models.Tree_Ensemble_Model(
        input_shape=X_train.shape[1], 
        output_shape=1, 
        model_type='random_forest', 
        task='regression',
        n_trees=100, 
        max_depth=10, 
        min_samples_split=5
    )
    XG_BOOST = baseline_models.Tree_Ensemble_Model(
        model_type='xgboost', 
        task='regression',
        n_trees=100, 
        learning_rate=0.01
    )

    return LASSO, RIDGE, LR, RANDOM_FOREST, XG_BOOST

def train(stock, start, end, test_ratio):
    LASSO, RIDGE, LR, RANDOM_FOREST, XG_BOOST = initialize(stock, start, end, test_ratio)

    # training
    LASSO = baseline_model_trainer.Trainer(stock, start, end, test_ratio, LASSO)
    LASSO = LASSO.train_model()

    RIDGE = baseline_model_trainer.Trainer(stock, start, end, test_ratio, RIDGE)
    RIDGE = RIDGE.train_model()

    LR = baseline_model_trainer.Trainer(stock, start, end, test_ratio, LR)
    LR = LR.train_model()

    RANDOM_FOREST = baseline_model_trainer.Trainer(stock, start, end, test_ratio, RANDOM_FOREST)
    RANDOM_FOREST = RANDOM_FOREST.train_model()

    XG_BOOST = baseline_model_trainer.Trainer(stock, start, end, test_ratio, XG_BOOST)
    XG_BOOST = XG_BOOST.train_model()

    return LASSO, RIDGE, LR, RANDOM_FOREST, XG_BOOST


#####################################################################
# This method executes baseline strategies and generates statistics #
#####################################################################

def execute_baseline_strategies(test_df, models, scaler, CFG):
    LASSO, RIDGE, LR, RANDOM_FOREST, XG_BOOST = models
    drop_cols = [CFG.target_col] + CFG.other_targets

    # Run baseline strategies
    naive_results = baseline_strategies.Strategies.naive_follow(test_df.copy())
    momentum_results = baseline_strategies.Strategies.momentum_strategy(test_df.copy())
    buy_and_hold_results = baseline_strategies.Strategies.buy_and_hold(test_df.copy())

    # Run model-based strategies
    lasso_results = baseline_strategies.Strategies.model_based_naive(model=LASSO, test_df=test_df.copy(), scaler=scaler, drop_columns=drop_cols)
    ridge_results = baseline_strategies.Strategies.model_based_naive(model=RIDGE, test_df=test_df.copy(), scaler=scaler, drop_columns=drop_cols)
    lr_results = baseline_strategies.Strategies.model_based_naive(model=LR, test_df=test_df.copy(), scaler=scaler, drop_columns=drop_cols)
    rf_results = baseline_strategies.Strategies.model_based_naive(model=RANDOM_FOREST, test_df=test_df.copy(), scaler=scaler, drop_columns=drop_cols)
    xgb_results = baseline_strategies.Strategies.model_based_naive(model=XG_BOOST, test_df=test_df.copy(), scaler=scaler, drop_columns=drop_cols)

    # Collect results
    all_results = {
        "Naive Follow": naive_results,
        "Momentum": momentum_results,
        "Buy and Hold": buy_and_hold_results,
        "LASSO Model": lasso_results,
        "RIDGE Model": ridge_results,
        "Linear Regression": lr_results,
        "Random Forest": rf_results,
        "XGBoost": xgb_results
    }
    
    return all_results