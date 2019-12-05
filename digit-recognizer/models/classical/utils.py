#!/usr/bin/python
import pandas as pd
import numpy as np
from tabulate import tabulate

def validate(data):
  return(data.isnull().any().any())

def normalize(data):
  normalized = data / 255.0
  return(normalized)

def printDatasets(X_train, Y_train, X_val, Y_val, X_test):
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