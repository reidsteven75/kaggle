import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate

DIR_DATA = './data/'
DIR_ARTIFACTS = './artifacts/'

TRAINING_EPOCHS = 15
VAL_TRAIN_RATIO = 0.3  # VAL / TEST

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

def decode(datum):
  return np.argmax(datum)

def decode_predictions(predictions):
  decoded = []
  for i in range(predictions.shape[0]):
    decoded_datum = decode(predictions[i])
    decoded.append(decoded_datum)
  return(decoded)

def debug_predictions(predictions):
  for i in range(predictions.shape[0]):
    datum = predictions[i]
    print('index: %d' % i)
    print('encoded datum: %s' % datum)
    decoded_datum = decode(predictions[i])
    print('decoded datum: %s' % decoded_datum)
    print()

def validate(data):
  return(data.isnull().any().any())

def normalize(data):
  normalized = data / 255.0
  return(normalized)

if __name__ == '__main__':

  # Load Data
  # ---------
  print('~ loading data ~')
  train = pd.read_csv(DIR_DATA + 'train.csv')
  test = pd.read_csv(DIR_DATA + 'test.csv')

  # Validate Data
  # -------------
  print('~ validating data ~')
  if (validate(train) == True or validate(test) == True):
    print('dataset(s) not formatted correctly')
    quit()

  # Train / Val / Test Datasets
  # ---------------------------
  print('~ prepairing datasets ~')
  Y_train = train.pop('label')
  X_train = train
  X_test = test

  # Normalize
  X_train = normalize(X_train)
  X_test = normalize(X_test)

  # Reshape
  Y_train = tf.keras.utils.to_categorical(Y_train)
  X_train = X_train.values.reshape(-1,28,28,1)
  X_test = test.values.reshape(-1,28,28,1)
  
  # Generate Validation Dataset
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VAL_TRAIN_RATIO, random_state=2)
  print_dataset_stats(X_train, Y_train, X_val, Y_val, X_test)

  # Model
  # -----
  print('~ compiling model ~')
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  print('input shape: ', model.input_shape)
  print('output shape: ', model.output_shape)

  model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  print('~ training model ~')
  model.fit(np.array(X_train), np.array(Y_train), epochs=TRAINING_EPOCHS)
  
  # Validate Model Performance
  # --------------------------
  print('~ validating model performance ~')
  model.evaluate(np.array(X_val), np.array(Y_val), verbose=2)

  # Predict
  # -------
  print('~ predicting ~')
  predictions = decode_predictions(model.predict(np.array(X_test)))
  submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})

  # Generate Artifact(s)
  # --------------------
  print('~ results (head) ~')
  print(submissions.head())
  submissions.to_csv(DIR_ARTIFACTS + 'predictions-neural-net.csv', index=False, header=True)