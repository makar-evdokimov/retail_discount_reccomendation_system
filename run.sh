# Final Project Pipeline (Group 9)

# activate virtualenv and change path to repo
source $PATH_ENV/bin/activate
cd $PATH_REPO

# install the necessary packages
pip install –-upgrade pip
pip install -r $PATH_REPO/final_project/requirements.txt 

# prepare data
python -m final_project.data

# train a model
python -m final_project.model_training

# build baseline models and do validation and benchmark evaluation
python -m final_project.benchmark_validation

# use the developed model to optimize future coupon assignments
python -m final_project.pred_opt

