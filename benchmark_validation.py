"""Building of baseline models, cross-validation and result benchmarking

This script uses time-based split of dataset, trains our model and
some alternative models (baselines) on the training set with
the further measurement of performance on the validation set

The cross-validated scores of performance metrics
are written in yaml file (scores.yaml)

Script should be run after data.py
"""

import os
import yaml
import lightgbm
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import sklearn.metrics
import final_project.data as lib
import numpy as np
import pandas as pd
import random

if __name__ == "__main__":

    # read config
    config = lib.read_yaml("config.yaml")
    path_data = config["path"]
    path_results = f"{path_data}/results"
    os.makedirs(path_results, exist_ok=True)

    # load the processed data, lists for model-specific indexing
    # and saved model hyperparameters

    train_set_cat = pd.read_parquet(f"{path_data}/processed/train_set_cat.pt")
    train_set_prod = pd.read_parquet(f"{path_data}/processed/train_set_prod.pt")

    y_cat = pd.DataFrame(train_set_cat['y_category'])
    y_prod = pd.DataFrame(train_set_prod['y_product'])

    lightgbm_cat_params = config["model"]["cat_level"]
    lightgbm_prod_params = config["model"]["prod_level"]

    order_features_cat = ['avg_cat_price', 'disc_self_cat', 'disc_subst', 'disc_compl',
                          'time_since_last_purchase_of_category', 'avg_purch_freq_shift', 'redemption_rate_shift']
    order_features_prod = ['max_price', 'discount', 'disc_other_in_cat', 'time_since_last_purchase_of_product',
                           'avg_purch_freq_shift', 'redemption_rate_shift', 'y_category']
    lgb_validation = ['max_price', 'discount', 'disc_other_in_cat', 'time_since_last_purchase_of_product',
                      'avg_purch_freq_shift', 'redemption_rate_shift', 'y_cat_pred_lgb']
    logit_validation = ['max_price', 'discount', 'disc_other_in_cat', 'time_since_last_purchase_of_product',
                        'avg_purch_freq_shift', 'redemption_rate_shift', 'y_cat_pred_logit']
    random_validation = ['max_price', 'discount', 'disc_other_in_cat', 'time_since_last_purchase_of_product',
                         'avg_purch_freq_shift', 'redemption_rate_shift', 'y_cat_random']

    product_features = ['max_price', 'discount', 'time_since_last_purchase_of_product', 'avg_purch_freq_shift',
                        'redemption_rate_shift']

    all_features = ['max_price', 'discount', 'disc_other_in_cat', 'time_since_last_purchase_of_product',
                    'avg_purch_freq_shift', 'redemption_rate_shift', 'disc_subst', 'disc_compl']

    # time-based train-test split for the product-level dataset and the category-level dataset

    X_train_cat, X_test_cat, y_train_cat, y_test_cat = sklearn.model_selection.train_test_split(
        train_set_cat,
        y_cat,
        test_size=0.2, shuffle=False
    )

    X_train_prod, X_test_prod, y_train_prod, y_test_prod = sklearn.model_selection.train_test_split(
        train_set_prod,
        y_prod,
        test_size=0.2, shuffle=False
    )


    logit_classifier = LogisticRegression(max_iter=100_000)
    lightgbm_classifier_cat = lightgbm.LGBMClassifier(**lightgbm_cat_params)
    lightgbm_classifier_prod = lightgbm.LGBMClassifier(**lightgbm_prod_params)

    # measures the performance of our lightGBM category-level model
    lightgbm_classifier_cat.fit(
        X_train_cat[order_features_cat], np.ravel(y_train_cat)
    )
    y_cat_pred_prob_lgb = lightgbm_classifier_cat.predict_proba(X_test_cat[order_features_cat])[:, 1]
    y_cat_pred_lgb = lightgbm_classifier_cat.predict(X_test_cat[order_features_cat])
    auc_cat_lgb = sklearn.metrics.roc_auc_score(y_test_cat, y_cat_pred_prob_lgb)
    # important: predictions by the category-level model are used
    # for the validation of product-level model
    X_test_cat['y_cat_pred_lgb'] = y_cat_pred_lgb

    # baseline for category-level model - logit model
    logit_classifier_cat = logit_classifier.fit(
        X_train_cat[order_features_cat], np.ravel(y_train_cat)
    )
    y_cat_pred_prob_logit = logit_classifier_cat.predict_proba(X_test_cat[order_features_cat])[:, 1]
    y_cat_pred_logit = logit_classifier_cat.predict(X_test_cat[order_features_cat])
    auc_cat_logit = sklearn.metrics.roc_auc_score(y_test_cat, y_cat_pred_prob_logit)
    # logit predictions are also used as an input for logit product-level model
    # to obtain fair cross-validated assessment of performance
    X_test_cat['y_cat_pred_logit'] = y_cat_pred_logit

    # random guess of category purchase probabilities
    rnd_guess_cat = np.random.uniform(0, 1, y_test_cat.shape[0])
    auc_cat_rnd = sklearn.metrics.roc_auc_score(y_test_cat, rnd_guess_cat)
    # it is alternatively used as an input for our product-level model
    X_test_cat['y_cat_random'] = np.rint(rnd_guess_cat)

    # category-level predictions from different models are transferred to product-level dataset
    X_test_prod = X_test_prod.merge(X_test_cat[['week', 'shopper', 'category', 'y_cat_pred_logit', 'y_cat_pred_lgb',
                                                'y_cat_random']], how='left', on=['week', 'shopper', 'category'])

    # measures the performance of our lightGBM product-level model
    # that uses the predictions from category level as an input
    lightgbm_classifier_prod.fit(
        X_train_prod[order_features_prod], np.ravel(y_train_prod)
    )
    y_prod_pred_prob_lgb = lightgbm_classifier_prod.predict_proba(X_test_prod[lgb_validation])[:, 1]
    bs_prod_lgb = sklearn.metrics.brier_score_loss(y_test_prod, y_prod_pred_prob_lgb)
    auc_prod_lgb = sklearn.metrics.roc_auc_score(y_test_prod, y_prod_pred_prob_lgb)

    # measures the performance of logit product-level model
    # that uses the logit predictions from category level as an input
    logit_classifier_prod = logit_classifier.fit(
        X_train_prod[order_features_prod], np.ravel(y_train_prod)
    )
    y_prod_pred_prob_logit = logit_classifier_prod.predict_proba(X_test_prod[logit_validation])[:, 1]
    bs_prod_logit = sklearn.metrics.brier_score_loss(y_test_prod, y_prod_pred_prob_logit)
    auc_prod_logit = sklearn.metrics.roc_auc_score(y_test_prod, y_prod_pred_prob_logit)

    # random guess of product purchase probabilities
    y_prod_rnd = np.random.uniform(0, 1, y_test_prod.shape[0])
    bs_prod_rnd = sklearn.metrics.brier_score_loss(y_test_prod, y_prod_rnd)
    auc_prod_rnd = sklearn.metrics.roc_auc_score(y_test_prod, y_prod_rnd)

    # our model predicts product purchase probabilities
    # given an input that includes random guess of category purchase probabilities
    # instead of predictions from category-level model
    y_prod_pred_perm = lightgbm_classifier_prod.predict_proba(X_test_prod[random_validation])[:, 1]
    bs_prod_perm = sklearn.metrics.brier_score_loss(y_test_prod, y_prod_pred_perm)
    auc_prod_perm = sklearn.metrics.roc_auc_score(y_test_prod, y_prod_pred_perm)

    # our model predicts product purchase probabilities
    # given an input that includes true information about category purchases
    # (which is not available for unlabeled data)
    y_prod_pred_true = lightgbm_classifier_prod.predict_proba(X_test_prod[order_features_prod])[:, 1]
    bs_prod_true = sklearn.metrics.brier_score_loss(y_test_prod, y_prod_pred_true)
    auc_prod_true = sklearn.metrics.roc_auc_score(y_test_prod, y_prod_pred_true)

    # logit model that is trained on the input containing only product-specific features
    # no information about categorical structure is used
    logit_classifier_prod_2 = logit_classifier.fit(
        X_train_prod[product_features], np.ravel(y_train_prod)
    )
    y_prod_pred_logit_2 = logit_classifier_prod_2.predict_proba(X_test_prod[product_features])[:, 1]
    bs_prod_logit_2 = sklearn.metrics.brier_score_loss(y_test_prod, y_prod_pred_logit_2)
    auc_prod_logit_2 = sklearn.metrics.roc_auc_score(y_test_prod, y_prod_pred_logit_2)

    X_train_prod = X_train_prod.merge(X_train_cat[['week', 'shopper', 'category', 'disc_subst', 'disc_compl']],
                                      how='left', on=['week', 'shopper', 'category'])
    X_train_prod.drop('y_category', axis=1, inplace=True)
    X_test_prod = X_test_prod.merge(X_test_cat[['week', 'shopper', 'category', 'disc_subst', 'disc_compl']], how='left',
                                    on=['week', 'shopper', 'category'])
    X_test_prod.drop('y_category', axis=1, inplace=True)

    # improved lightgbm model (one-level) that uses both product-specific features
    # and the features that capture cross-category discount effects
    # Shows better performance on validation set but not used further
    lightgbm_classifier_fin = lightgbm.LGBMClassifier(**lightgbm_prod_params)
    lightgbm_classifier_fin.fit(
        X_train_prod[all_features], np.ravel(y_train_prod),
    )
    y_prod_pred_fin = lightgbm_classifier_fin.predict_proba(X_test_prod[all_features])[:, 1]
    bs_prod_fin = sklearn.metrics.brier_score_loss(y_test_prod, y_prod_pred_fin)
    auc_prod_fin = sklearn.metrics.roc_auc_score(y_test_prod, y_prod_pred_fin)

    # AUC scores for category-level models and AUC scores and Brier scores
    # for product-level models are saved in yaml file
    scores = {
        "category_level": {
            "lightgbm_auc": float(auc_cat_lgb),
            "logit_auc": float(auc_cat_logit),
            "random_auc": float(auc_cat_rnd),
        },
        "product_level": {
            "lightgbm": {
                "auc": float(auc_prod_lgb),
                "brier_score": float(bs_prod_lgb),
            },
            "logit": {
                "auc": float(auc_prod_logit),
                "brier_score": float(bs_prod_logit),
            },
            "random": {
                "auc": float(auc_prod_rnd),
                "brier_score": float(bs_prod_rnd),
            },
            "lightgbm_random_input": {
                "auc": float(auc_prod_perm),
                "brier_score": float(bs_prod_perm),
            },
            "logit_product_features_only": {
                "auc": float(auc_prod_logit_2),
                "brier_score": float(bs_prod_logit_2),
            },
            "improved_lightgbm": {
                "auc": float(auc_prod_fin),
                "brier_score": float(bs_prod_fin),
            },
        },
    }
    lib.write_yaml(scores, f"{path_results}/model_scores.yaml")
