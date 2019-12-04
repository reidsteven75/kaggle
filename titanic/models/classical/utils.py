#!/usr/bin/python

import math
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tabulate import tabulate

def printScoreResultsKFold(model_score_results):
  header = ['MODEL', 'ACCURACY (avg)', 'VARIANCE (max-min)']
  table = []
  for result in model_score_results:
    accuracy = np.average(result['score'])
    variance = np.max(result['score']) - np.min(result['score'])
    table.append([result['model'], accuracy, variance])
  print(tabulate(table, headers=header, tablefmt='fancy_grid'))

def printScoreResultsTrainVal(model_score_results, score_type=''):
  header = ['MODEL', 'ACCURACY ' + '(' + score_type + ')']
  table = []
  for result in model_score_results:
    table.append([result['model'], result['score']])
  print(tabulate(table, headers=header, tablefmt='fancy_grid'))

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

def visualizeCorrelations(dataset, y, threshold, num):
  correlation = dataset.select_dtypes(exclude='object').corr()
  if threshold:
    sns.heatmap(data=correlation>threshold, cmap='YlGnBu', annot=True)
  else:
    sns.heatmap(correlation, cmap='YlGnBu', annot=True)
  plt.show()
  top_correlated_features = correlation[y].sort_values(ascending=False).head(num)
  print(top_correlated_features)

def visualizeFeatureCorrelation(visualization, feature, target, dataset):
  if (visualization == 'factorplot'):
    g = sns.factorplot(x=feature,y=target, data=dataset, kind='bar', size=6, palette='muted')
    g.despine(left=True)
    g = g.set_ylabels('Target')
    plt.show()
  elif (visualization == 'facetgrid'):
    g = sns.FacetGrid(dataset, col=target)
    g = g.map(sns.distplot, feature)
    plt.show()
  elif (visualization == 'barplot'):
    g = sns.barplot(x=feature, y=target, data=dataset)
    plt.show()

def visualizeSkew(dataset, feature):
  y = dataset[feature]
  sns.distplot(y)
  plt.show()
