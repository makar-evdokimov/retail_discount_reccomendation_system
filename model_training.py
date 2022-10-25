"""Hyperparameter tuning and model training

This script tunes LightGBM hyperparameters with the use
of Optuna library (this part of the code is presented as a comment)
and trains the two models on the processed input data using the configuration
of tuned hyperparameters.

Script must be run after data.py or using backup files
x_cat.pt, x_prod.pt, y_cat.pt, y_prod.pt
"""

import sklearn
import sklearn.model_selection
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import sklearn.metrics
import sklearn.datasets

import optuna
import lightgbm
import joblib
import final_project.data as lib
import os
import pandas as pd
import numpy as np
import yaml

# commented functions below are used by Optuna algorithm
# it samples a configuration of hyperparameters
# and calculates the cross-validated value of accuracy
# for the corresponding model


# def objective_cat(trial):
#     """ Function that is optimized to tune hyperparameters of category-level model
#
#     Uses AUC as metric and the value of the function is predictive accuracy
#     """
#
#     data, target = X, y
#     train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, shuffle=False)
#     dtrain = lightgbm.Dataset(train_x, label=train_y)

#     param = {
#         'objective': 'binary',
#         'metric': 'auc',
#         "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#     }

#    gbm = lightgbm.train(param, dtrain)
#    preds = gbm.predict(test_x)
#    pred_labels = np.rint(preds)
#    accuracy = sklearn.metrics.accuracy_score(test_y, pred_labels)
#    return accuracy


# def objective_prod(trial):
#     """ Function that is optimized to tune hyperparameters of product-level model
#
#     Uses binary log-loss as metric and the value of the function is F-score
#     """
#     data, target = X, y
#     train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2, shuffle=False)
#     dtrain = lightgbm.Dataset(train_x, label=train_y)

#     param = {
#         'objective': 'binary',
#         'metric': 'binary_logloss',
#         "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
#         'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#         'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#         'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#         'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#         'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#         'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#     }

#     gbm = lightgbm.train(param, dtrain)
#     preds = gbm.predict(test_x)
#     pred_labels = np.rint(preds)
#     f1_score = sklearn.metrics.f1_score(test_y, pred_labels)
#     return f1_score

# Below the code part that conducts the tuning process
# for both model levels is presented

# if __name__ == "__main__":
#     X = X_cat
#     y = y_cat
#     study_cat_without_pruning = optuna.create_study(direction="maximize")
#     study_cat_without_pruning.optimize(objective_cat, n_trials=250)

#     print("Number of finished trials: {}".format(len(study_cat_without_pruning.trials)))

#     print("Best trial:")
#     trial = study_cat_without_pruning.best_trial

#     print("  Value: {}".format(trial.value))

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))

#     lightgbm_cat_params = study_cat_without_pruning.best_params

# if __name__ == "__main__":
#     X = X_prod
#     y = y_prod
#     study_prod_without_pruning = optuna.create_study(direction="maximize")
#     study_prod_without_pruning.optimize(objective_prod, n_trials=250)         # In practice 100 trials were conducted

#     print("Number of finished trials: {}".format(len(study_prod_without_pruning.trials)))

#     print("Best trial:")
#     trial = study_prod_without_pruning.best_trial

#     print("  Value: {}".format(trial.value))

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))

#     lightgbm_prod_params = study_prod_without_pruning.best_params

# Two last commented procedures below are the tuning of two models' parameters
# with LightGBM Tuner, Optuna's framework-specific algorithm, which
# has also been considered as a tuning option

# import optuna.integration.lightgbm as lgb

# if __name__ == "__main__":
#      data, target = X_cat, y_cat
#      dtrain = lgb.Dataset(data, label=target)

#      params = {
#          'objective': 'binary',
#          'metric': 'auc',
#          "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
#          'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#          'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#          'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#          'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#          'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#          'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#          'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#      }

#      tuner_cat = lgb.LightGBMTunerCV(
#          params, dtrain, verbose_eval=100, early_stopping_rounds=100, folds=TimeSeriesSplit(n_splits=5)
#      )

#      tuner_cat.run()

#      print("Best score:", tuner_cat.best_score)
#      best_params = tuner_cat.best_params
#      print("Best params:", best_params)
#      print("  Params: ")
#      for key, value in best_params.items():
#          print("    {}: {}".format(key, value))

#      lightgbm_cat_params = tuner_cat.best_params

# if __name__ == "__main__":
#      data, target = X_prod, y_prod
#      dtrain = lgb.Dataset(data, label=target)

#      params = {
#          'objective': 'binary',
#          'metric': 'binary_logloss',
#          "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
#          'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
#          'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
#          'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#          'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
#          'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
#          'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
#          'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#      }

#      tuner_prod = lgb.LightGBMTunerCV(
#          params, dtrain, verbose_eval=100, early_stopping_rounds=100, folds=TimeSeriesSplit(n_splits=5)
#      )

#      tuner_prod.run()

#      print("Best score:", tuner_prod.best_score)
#      best_params = tuner_prod.best_params
#      print("Best params:", best_params)
#      print("  Params: ")
#      for key, value in best_params.items():
#          print("    {}: {}".format(key, value))

#     lightgbm_prod_params = tuner_prod.best_params

if __name__ == "__main__":

    # read config
    config = lib.read_yaml("config.yaml")
    path_data = config["path"]
    path_models = f"{path_data}/models"
    os.makedirs(path_models, exist_ok=True)

    # load the processed data
    X_cat = pd.read_parquet(f"{path_data}/processed/x_cat.pt")
    y_cat = pd.read_parquet(f"{path_data}/processed/y_cat.pt")
    X_prod = pd.read_parquet(f"{path_data}/processed/x_prod.pt")
    y_prod = pd.read_parquet(f"{path_data}/processed/y_prod.pt")

    # parameters have been already tuned by Optuna algorithm
    # and saved to config
    lightgbm_cat_params = config["model"]["cat_level"]
    lightgbm_prod_params = config["model"]["prod_level"]

    # both models are trained with specific parameter sets
    lightgbm_classifier_cat = lightgbm.LGBMClassifier(**lightgbm_cat_params)
    lightgbm_classifier_cat.fit(
        X_cat, np.ravel(y_cat)
    )

    lightgbm_classifier_prod = lightgbm.LGBMClassifier(**lightgbm_prod_params)
    lightgbm_classifier_prod.fit(
        X_prod, np.ravel(y_prod)
    )

    # save the models
    joblib.dump(lightgbm_classifier_cat, f"{path_models}/lightgbm_cat_level_model.pkl")
    joblib.dump(lightgbm_classifier_prod, f"{path_models}/lightgbm_prod_level_model.pkl")
