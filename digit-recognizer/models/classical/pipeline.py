#!/usr/bin/python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import utils

DIR_DATA = './data/'
DIR_ARTIFACTS = './artifacts/'
ARTIFACT_NAME = 'predictions.csv'

# Load Data
# ---------
print('~ loading data ~')
train = pd.read_csv(DIR_DATA + 'train.csv')
test = pd.read_csv(DIR_DATA + 'test.csv')

# Validate Data
# -------------
print('~ validating data ~')
if (utils.validate(train) == True or utils.validate(test) == True):
  print('dataset(s) not formatted correctly')
  quit()

# Visually Inspect Data
# ---------------------
# X_train = X_train.reshape(X_train.shape[0], 28, 28)
# for i in range(6, 9):
#   renderDigitImage(i, X_train[i], y_train[i])

# Train / Test Datasets
# ---------------------
print('~ prepairing datasets ~')
X_train = (train.iloc[:,1:].values).astype('float32') 
Y_train = train.iloc[:,0].values.astype('int32')
X_test = test.values.astype('float32')

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.3, random_state=2)
utils.printDatasets(X_train, Y_train, X_val, Y_val, X_test)

# Normalize Data
# --------------
print('~ normalizing ~')
X_train = utils.normalize(X_train)
X_val = utils.normalize(X_val)
X_test = utils.normalize(X_test)

# Model
# -----
print('~ training test/val split model(s) ~')
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,Y_train)

print('~ scoring test/val split model(s) ~')
Y_pred=model.predict(X_val)
print('Accuracy:', metrics.accuracy_score(Y_val, Y_pred))

# Predict
# -------
print('~ predicting ~')
predictions = model.predict(X_test)
submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
submissions.to_csv(DIR_ARTIFACTS + ARTIFACT_NAME, index=False, header=True)
print('Artifact file generated:')
print(ARTIFACT_NAME)
print('~ done ~')