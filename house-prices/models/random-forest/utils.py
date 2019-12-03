#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

def print_dataset_stats(X_train, Y_train, X_val, Y_val, X_test):
  count_train = np.size(X_train, 0)
  count_val = np.size(X_val, 0)
  count_test = np.size(X_test, 0)

  count_total = count_train + count_val + count_test

  header = ['DATASET', '# SAMPLES', '% DISTRIBUTION']
  table = [
    ['Train', f'{count_train:,}', count_train / count_total],
    ['Val', f'{count_val:,}', count_val / count_total],
    ['Test', f'{count_test:,}', count_test / count_total]
  ]
  print(tabulate(table, headers=header, tablefmt='fancy_grid'))

  header = ['DATASET', 'SHAPE']
  table = [
    ['X_train', X_train.shape],
    ['Y_train', Y_train.shape],
    ['X_val',X_val.shape],
    ['Y_val', Y_val.shape],
    ['X_test', X_test.shape]
  ]
  print(tabulate(table, headers=header, tablefmt='fancy_grid'))

def inspect(data):
  print(data.isna().sum().sort_values(ascending=False).head(20))

def dropOutliers(dataset, feature, threshold):
  return(dataset.drop(dataset[ dataset[feature] > threshold ].index, inplace=True))

def outlierAnalysis(dataset, target):
  y = dataset[target]
  num_attributes = dataset.select_dtypes(exclude='object').drop(target, axis=1).copy()
  f = plt.figure(figsize=(12,20))
  numPlots = len(num_attributes.columns)
  for i in range(numPlots):
    numColumns = 5
    numRows = math.ceil(numPlots/numColumns)
    f.add_subplot(numRows, numColumns, i+1)
    sns.scatterplot(num_attributes.iloc[:,i], y)

  plt.tight_layout(h_pad=1.0)
  plt.show()

def visualize_correlations(dataset, y, num):
  correlation = dataset.select_dtypes(exclude='object').corr()
  sns.heatmap(data=correlation>0.80, cmap='YlGnBu', annot=True)
  plt.show()
  top_correlated_features = correlation[y].sort_values(ascending=False).head(num)
  print(top_correlated_features)

def visualize_skew(dataset, target):
  y = dataset[target]
  sns.distplot(y)
  plt.title('Raw')
  plt.show()
  sns.distplot(np.log(y))
  plt.title('Log')
  plt.show()
