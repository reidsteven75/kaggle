#!/usr/bin/python
import math
import pandas as pd
import numpy as np
import featuretools as ft
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # handling missing variables: categorical, numerical
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from tabulate import tabulate

import utils

# MODELS
# ======
# Random forest
# Linear regression
# Logistic regression

# FEATURE ENGINEERING
# ===================
# Features with a high percentage of missing values
# Collinear (highly correlated) features
# Features with zero importance in a tree-based model
# Features with low importance
# Features with a single unique value

# PERFORMANCE INCREASE IDEAS
# ==========================
# see if imputation helps improve accuracy
# see if automl can be used for any of this

dir_data = '../../data/'
dir_artifacts = '../../artifacts/'

# dir_data = './data/'
# dir_artifacts = './artifacts/'

VAL_TRAIN_RATIO = 0.3  # VAL / TEST
NUM_CORRELATIONS = 15

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
# visualize_skew(train, 'SalePrice')
train['SalePrice'] = np.log(train['SalePrice'])

# Correlations
# ------------
print('~ correlation analysis ~')
# visualize_correlations(train, 'SalePrice', NUM_CORRELATIONS)

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
# outlierAnalysis(train, 'SalePrice')
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

# Generate Validation Dataset
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VAL_TRAIN_RATIO, random_state=2)
utils.print_dataset_stats(X_train, Y_train, X_val, Y_val, X_test)

# Model
# -----
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train,Y_train)

# Validate
# -------
print('Accuracy:', model.score(X_val, Y_val))

# Predict
# -------
predictions = model.predict(X_test)
submissions=pd.DataFrame({'Id': list(range(1,len(predictions)+1)), 'SalePrice': predictions})
submissions.to_csv(dir_artifacts + 'predictions-random-forest.csv', index=False, header=True)