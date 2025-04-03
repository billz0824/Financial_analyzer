from Data import processor
from Models import baseline_models, baseline_strategies, baseline_model_trainer
import matplotlib.pyplot as plt

######################################################
# This module initializes and trains baseline models #
######################################################

def initialize(stock, start, end, test_ratio):
    _, X_train, _, _, _, testing, scaler = processor.baseline_preprocessing(stock, start, end, test_ratio=test_ratio)
    LASSO = baseline_models.Linear_Model(input_shape=X_train.shape[1], output_shape=1, model_type='lasso', alpha=0.2)
    RIDGE = baseline_models.Linear_Model(input_shape=X_train.shape[1], output_shape=1, model_type='ridge', alpha=0.2)
    LR = baseline_models.Linear_Model(input_shape=X_train.shape[1], output_shape=1)
    RANDOM_FOREST = baseline_models.Tree_Ensemble_Model(
        input_shape=X_train.shape[1], 
        output_shape=1, 
        model_type='random_forest', 
        task='regression',
        n_trees=50, 
        max_depth=None, 
        min_samples_split=2
    )
    XG_BOOST = baseline_models.Tree_Ensemble_Model(
        model_type='xgboost', 
        task='regression',
        n_trees=50, 
        learning_rate=0.01
    )

    return LASSO, RIDGE, LR, RANDOM_FOREST, XG_BOOST, testing, scaler

def train(stock, start, end, test_ratio, selected_models):
    LASSO, RIDGE, LR, RANDOM_FOREST, XG_BOOST, testing, scaler = initialize(stock, start, end, test_ratio)

    # train selected models
    lasso_trainer = ridge_trainer = lr_trainer = rf_trainer = xgb_trainer = None

    if "LASSO" in selected_models:
        lasso_trainer = baseline_model_trainer.Trainer(stock, start, end, test_ratio, LASSO)
        lasso_trainer.train_model()
    if "RIDGE" in selected_models:
        ridge_trainer = baseline_model_trainer.Trainer(stock, start, end, test_ratio, RIDGE)
        ridge_trainer.train_model()
    if "LR" in selected_models:
        lr_trainer = baseline_model_trainer.Trainer(stock, start, end, test_ratio, LR)
        lr_trainer.train_model()
    if "RANDOM_FOREST" in selected_models:
        rf_trainer = baseline_model_trainer.Trainer(stock, start, end, test_ratio, RANDOM_FOREST)
        rf_trainer.train_model()
    if "XG_BOOST" in selected_models:
        xgb_trainer = baseline_model_trainer.Trainer(stock, start, end, test_ratio, XG_BOOST)
        xgb_trainer.train_model()

    return {
        "LASSO": lasso_trainer,
        "RIDGE": ridge_trainer,
        "LR": lr_trainer,
        "RANDOM_FOREST": rf_trainer,
        "XG_BOOST": xgb_trainer,
        "Test_Df": testing,
        "Scaler": scaler
    }

#####################################################################
# This method executes baseline strategies and generates statistics #
#####################################################################

def execute_baseline_strategies(stock, start, end, test_ratio, selected_models=None):
    config = processor.Baseline_CFG
    trainers = train(stock, start, end, test_ratio, selected_models=selected_models)
    drop_cols = [config.target_col] + config.other_targets

    test_df = trainers["Test_Df"]
    scaler = trainers["Scaler"]

    strats = baseline_strategies.Strategies()

    # Baseline strategies
    all_results = {
        "Naive Follow": strats.naive_follow(test_df.copy()),
        "Momentum": strats.momentum_strategy(test_df.copy()),
        "Buy and Hold": strats.buy_and_hold(test_df.copy()),
    }

    # Model-based strategies (Conditional on user preference)
    if "LASSO" in selected_models:
        all_results["LASSO Model"] = strats.model_based_naive(trainers["LASSO"].model, test_df.copy(), scaler, drop_cols)
    if "RIDGE" in selected_models:
        all_results["RIDGE Model"] = strats.model_based_naive(trainers["RIDGE"].model, test_df.copy(), scaler, drop_cols)
    if "LR" in selected_models:
        all_results["Linear Regression"] = strats.model_based_naive(trainers["LR"].model, test_df.copy(), scaler, drop_cols)
    if "RANDOM_FOREST" in selected_models:
        all_results["Random Forest"] = strats.model_based_naive(trainers["RANDOM_FOREST"].model, test_df.copy(), scaler, drop_cols)
    if "XG_BOOST" in selected_models:
        all_results["XGBoost"] = strats.model_based_naive(trainers["XG_BOOST"].model, test_df.copy(), scaler, drop_cols)

    return all_results