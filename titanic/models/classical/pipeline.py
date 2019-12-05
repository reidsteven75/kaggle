#!/usr/bin/python
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # handling missing variables: categorical, numerical
from sklearn.preprocessing import LabelEncoder 
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
# - married / not married
# - classify males as married if their last name matches a female with the same last name
# - classify name titles

DIR_DATA = './data/'
DIR_ARTIFACTS = './artifacts/'
ARTIFACT_NAME = 'predictions.csv'

TARGET = 'Survived'
VAL_TRAIN_RATIO = 0.3  # VAL / TEST
NUM_CORRELATIONS = 15
CROSS_VALIDATION_K_FOLDS = 10

print('~ loading data ~')

# Load Data
# ---------
train = pd.read_csv(DIR_DATA + 'train.csv', index_col=0)
test = pd.read_csv(DIR_DATA + 'test.csv', index_col=0)

all_data = pd.concat([train.drop(TARGET, axis=1), test])

# Correlations
# ------------
print('~ correlation analysis ~')
# utils.visualizeFeatureCorrelation(visualization='factorplot', feature='SibSp', target=TARGET, dataset=train)
# utils.visualizeFeatureCorrelation(visualization='factorplot', feature='Parch', target=TARGET, dataset=train)
# utils.visualizeFeatureCorrelation(visualization='factorplot', feature='Embarked', target=TARGET, dataset=train)
# utils.visualizeFeatureCorrelation(visualization='facetgrid', feature='Age', target=TARGET, dataset=train)
# utils.visualizeFeatureCorrelation(visualization='facetgrid', feature='Fare', target=TARGET, dataset=train)
# utils.visualizeFeatureCorrelation(visualization='barplot', feature='Sex', target=TARGET, dataset=train)
# utils.visualizeFeatureCorrelation(visualization='barplot', feature='Pclass', target=TARGET, dataset=train)
'''
- Fare = highest correlation with survival (0.26)
- SibSp & Parch decently correlated with eachother (0.41)
- SibSp, Parch, Embarked, Sex, Pclass seem to make a difference with survival
'''

# Clean, Encode Numerics & Categories
# ------------------------------------
print('~ cleaning & encoding ~')

# cateogrize names by titles into [ 'Mr', 'Lady', 'Master', 'Other' ]
all_data['title'] = pd.Series([i.split(',')[1].split('.')[0].strip() for i in all_data['Name']])
non_other_titles = ['Mrs', 'Miss', 'Ms', 'Mme', 'Mlle', 'Mr', 'Master']
all_data['title'] = all_data['title'].map(lambda s: 'Other' if s not in non_other_titles else s)
grouped_lady_titles = ['Mrs', 'Miss', 'Mme', 'Mlle', 'Ms']
ignore_titles = ['Mr', 'Master', 'Other']
all_data['title'] = all_data['title'].map(lambda s: 'Lady' if s in non_other_titles and s not in ignore_titles else s)
all_data = all_data.drop(['Name'], axis=1)

# classify by cabin group only instead of group+room
all_data['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in all_data['Cabin'] ])
all_data['Cabin'] = all_data['Cabin'].fillna('X')

# parse ticket prefix (the # is useless)
ticket_parsed = []
for i in list(all_data['Ticket']):
  if not i.isdigit():
    ticket_parsed.append(i.replace('.','').replace('/','').strip().split(' ')[0]) 
  else:
    ticket_parsed.append('X')
all_data['Ticket'] = ticket_parsed

# convert categorical
all_data = pd.get_dummies(all_data, columns=['title'], prefix='title')
all_data = pd.get_dummies(all_data, columns=['Embarked'], prefix='em')
all_data = pd.get_dummies(all_data, columns=['Ticket'], prefix='tx')
all_data = pd.get_dummies(all_data, columns=['Cabin'], prefix='cab')
all_data = pd.get_dummies(all_data, columns=['Sex'], prefix='sex')

# fill in any remaining null values
all_data = all_data.fillna(all_data.mean())

clean_train = pd.concat([all_data[:train.shape[0]], train[TARGET]], axis=1)

# Skew Analysis
# ------------
print('~ skew analysis ~')
# utils.visualizeSkew(clean_train, 'Fare')
clean_train['Fare'] = clean_train['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
# utils.visualizeSkew(clean_train, 'Fare')

# Outlier Analysis
# ----------------
print('~ outlier analysis ~')
# utils.outlierAnalysis(train, TARGET)

# Train / Val / Test Datasets
# ---------------------------
print('~ prepairing datasets ~')
X_test = all_data[train.shape[0]:]
Y_train = clean_train.pop(TARGET)
X_train = clean_train

Y_train_cross_val = Y_train
X_train_cross_val = X_train

# Generate Validation Dataset
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VAL_TRAIN_RATIO, shuffle=True, random_state=2)
utils.print_dataset_stats(X_train, Y_train, X_val, Y_val, X_test)

# Model
# -----
print('~ training test/val split model(s) ~')
model_random_forest = RandomForestClassifier().fit(X_train,Y_train)
model_gradient_boosting = GradientBoostingClassifier().fit(X_train,Y_train)
model_ada_boost = AdaBoostClassifier().fit(X_train,Y_train)
model_logistic_regression = LogisticRegression( max_iter=10000 ).fit(X_train,Y_train)

print('~ scoring test/val split model(s) ~')
model_score_results = [
  {'model': 'Random Forest Classifier', 'score': model_random_forest.score(X_val, Y_val)},
  {'model': 'Gradient Boosting Classifier', 'score': model_gradient_boosting.score(X_val, Y_val)},
  {'model': 'ADA Boost Classifier', 'score': model_ada_boost.score(X_val, Y_val)},
  {'model': 'Logistic Regression', 'score': model_logistic_regression.score(X_val, Y_val)}
]
utils.printScoreResultsTrainVal(model_score_results, score_type='R2')

print('~ training k-fold cross-validation model(s) ~')
model_random_forest = RandomForestClassifier().fit(X_train_cross_val,Y_train_cross_val)
model_gradient_boosting = GradientBoostingClassifier().fit(X_train_cross_val,Y_train_cross_val)
model_ada_boost = AdaBoostClassifier().fit(X_train_cross_val,Y_train_cross_val)
model_logistic_regression = LogisticRegression( max_iter=10000 ).fit(X_train_cross_val,Y_train_cross_val)

print('~ scoring k-fold cross-validation model(s) ~')
model_score_results = [
  {'model': 'Random Forest Classifier', 'score': cross_val_score(model_random_forest, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)},
  {'model': 'Gradient Boosting Classifier', 'score': cross_val_score(model_gradient_boosting, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)},
  {'model': 'ADA Boost Classifier', 'score': cross_val_score(model_ada_boost, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)},
  {'model': 'Logistic Regression', 'score': cross_val_score(model_logistic_regression, X_train_cross_val, Y_train_cross_val, cv=CROSS_VALIDATION_K_FOLDS)}
]
utils.printScoreResultsKFold(model_score_results)

# Validate
# --------
print('~ selecting model ~')
model = model_gradient_boosting

# Predict
# -------
print('~ predicting ~')
predictions = model.predict(X_test)
print('Results (first 10)')
print(predictions[:10])

# Create Artifact
# ---------------
print('~ generating artifacts ~')
submissions=pd.DataFrame({'PassengerId': X_test.index.values, TARGET: predictions})
submissions.to_csv(DIR_ARTIFACTS + ARTIFACT_NAME, index=False, header=True)
print('Artifact file generated:')
print(ARTIFACT_NAME)
print('~ done ~')