# Personalized Discount Recommendation System for a Retail Store 

This folder implements our solution for the project


## Pipeline

The pipeline includes the following steps:
1. Explore data, construct features, build datasets
1. Tune the hyperparameters
1. Train the models
1. Build baseline models and benchmark the results
1. Use the model for coupon assignment optimization


## Environment variables

Set the following environment variables before running the pipeline:

| Variable    | Description                                                            | Example          |
| ----------- | -----------------------------------------------------------------------| ---------------- |
| `PATH_REPO` | path of the repository/working directory to place the folder           | `$HOME/mlim`     |
| `PATH_ENV`  | path of the virtual environment                                        | `$HOME/env-mlim` |


Before running the pipeline, set the path for the folder with config.yaml as a working directory
and set the value of config['path'] in config.yaml as the path for the folder with files

PATH_REPO (working directory) should be te path to the folder that contatins this folder.
