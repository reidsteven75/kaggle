#!/usr/bin/python
import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # handling missing variables: categorical, numerical
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor

import utils

# Ignore unnecessary warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

# PERFORMANCE INCREASE IDEAS
# ==========================
# - see if imputation helps improve accuracy
# - hyper-parameter tuning

# PERFORMANCE VALIDATION IDEAS
# ============================
# - look into bias, variance library - http://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/

dir_data = '../../data/'
dir_artifacts = '../../artifacts/'

# dir_data = './data/'
# dir_artifacts = './artifacts/'

VAL_TRAIN_RATIO = 0.3  # VAL / TEST
NUM_CORRELATIONS = 15
CROSS_VALIDATION_K_FOLDS = 10

print('~ loading data ~')

# Load Data
# ---------
train = pd.read_csv(dir_data + 'train.csv', index_col=0)
test = pd.read_csv(dir_data + 'test.csv', index_col=0)

all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))

# print('Features: ' + str(len(train.columns)))

# Un-Skew (Log Transform)
# -----------------------
print('~ unskewing ~')
# utils.visualizeSkew(train, 'SalePrice')
train['SalePrice'] = np.log(train['SalePrice'])

# Correlations
# ------------
print('~ correlation analysis ~')
# utils.visualizeCorrelations(train, 'SalePrice', NUM_CORRELATIONS)

remove_columns = ['MiscVal', 'MSSubClass', 'MoSold', 'YrSold', 'GarageArea', 'GarageYrBlt', 'TotRmsAbvGrd']
all_data = all_data.drop(remove_columns, axis=1)

# Clean, Encode Numerics & Categories
# ------------------------------------
print('~ cleaning & encoding ~')

all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())

clean_train = pd.concat([all_data[:train.shape[0]], train.SalePrice], axis=1)

# Outlier Analysis
# ----------------
print('~ outlier analysis ~')
# utils.outlierAnalysis(train, 'SalePrice')
utils.dropOutliers(clean_train, 'LotFrontage', 220)
utils.dropOutliers(clean_train, 'LotArea', 110000)
utils.dropOutliers(clean_train, 'BsmtFinSF1', 2500)
utils.dropOutliers(clean_train, 'BsmtFinSF2', 3500)
utils.dropOutliers(clean_train, 'TotalBsmtSF', 3500)
utils.dropOutliers(clean_train, '1stFlrSF', 4000)

# Train / Val / Test Datasets
# ---------------------------
print('~ prepairing datasets ~')
X_test = all_data[train.shape[0]:]
Y_train = clean_train.pop('SalePrice')
X_train = clean_train

Y_train_cross_val = Y_train
X_train_cross_val = X_train

# Generate Validation Dataset
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VAL_TRAIN_RATIO, shuffle=True, random_state=2)
utils.print_dataset_stats(X_train, Y_train, X_val, Y_val, X_test)

# Model
# -----
print('~ training test/val split model(s) ~')
model_random_forest = RandomForestRegressor().fit(X_train,Y_train)
model_gradient_boosting_regressor = GradientBoostingRegressor().fit(X_train,Y_train)
model_ada_boost_regressor = AdaBoostRegressor().fit(X_train,Y_train)
model_xgb_boost = XGBRegressor(objective='reg:squarederror').fit(X_train,Y_train)

print('~ scoring test/val split model(s) ~')
model_score_results = [
  {'model': 'Random Forest Regressor', 'score': model_random_forest.score(X_val, Y_val)},
  {'model': 'Gradient Boosting Regressor', 'score': model_gradient_boosting_regressor.score(X_val, Y_val)},
  {'model': 'ADA Boost Regressor', 'score': model_ada_boost_regressor.score(X_val, Y_val)},
  {'model': 'XGB Boost Regressor', 'score': model_xgb_boost.score(X_val, Y_val)}
]
utils.printScoreResultsTrainVal(model_score_results, score_type='R2')

print('~ training k-fold cross-validation model(s) ~')
model_random_forest = RandomForestRegressor().fit(X_train_cross_val,Y_train_cross_val)
model_gradient_boosting_regressor = GradientBoostingRegressor().fit(X_train_cross_val,Y_train_cross_val)
model_ada_boost_regressor = AdaBoostRegressor().fit(X_train_cross_val,Y_train_cross_val)
model_xgb_boost = XGBRegressor(objective='reg:squarederror').fit(X_train_cross_val,Y_train_cross_val)

print('~ scoring k-fold cross-validation model(s) ~')
model_score_results = [
  {'model': 'Random Forest Regressor', 'score': cross_val_score(model_random_forest, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)},
  {'model': 'Gradient Boosting Regressor', 'score': cross_val_score(model_gradient_boosting_regressor, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)},
  {'model': 'ADA Boost Regressor', 'score': cross_val_score(model_ada_boost_regressor, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)},
  {'model': 'XGB Boost Regressor', 'score': cross_val_score(model_xgb_boost, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)}
]
utils.printScoreResultsKFold(model_score_results)

# Validate
# --------
print('~ selecting model ~')
model = model_gradient_boosting_regressor

# Predict
# -------
print('~ predicting ~')
predictions = model.predict(X_test)
predictions = np.exp(predictions) # reverse log function 
print('Results...')
print(predictions)

# Create Artifact
# ---------------
print('~ generating artifacts ~')
submissions=pd.DataFrame({'Id': X_test.index.values, 'SalePrice': predictions})
submissions.to_csv(dir_artifacts + 'predictions-random-forest.csv', index=False, header=True)

print('~ done ~')