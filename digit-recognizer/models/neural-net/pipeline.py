#!/usr/bin/python
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate

import utils

DIR_DATA = './data/'
DIR_ARTIFACTS = './artifacts/'

TRAINING_EPOCHS = 15
VAL_TRAIN_RATIO = 0.3  # VAL / TEST

# Load Data
# ---------
print('~ loading data ~')
train = pd.read_csv(DIR_DATA + 'train.csv', dtype='float64')
test = pd.read_csv(DIR_DATA + 'test.csv', dtype='float64')

# Validate Data
# -------------
print('~ validating data ~')
if (utils.validate(train) == True or utils.validate(test) == True):
  print('dataset(s) not formatted correctly')
  quit()

# Train / Val / Test Datasets
# ---------------------------
Y_train = train.pop('label')
X_train = train
X_test = test

# Normalize
print('~ normalizing ~')
X_train = utils.normalize(X_train)
X_test = utils.normalize(X_test)

# Reshape
print('~ reshaping ~')
Y_train = tf.keras.utils.to_categorical(Y_train)
X_train = X_train.values.reshape(-1,28,28,1)
X_test = test.values.reshape(-1,28,28,1)

# Generate Validation Dataset
print('~ prepairing datasets ~')
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=VAL_TRAIN_RATIO, random_state=2)
utils.printDatasets(X_train, Y_train, X_val, Y_val, X_test)

# Model
# -----
print('~ configuring & compiling model ~')
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(28,28,1)),
  tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5),padding='Same', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2)),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Flatten(),
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
print('~ scoring model ~')
model.evaluate(np.array(X_val), np.array(Y_val), verbose=2)

# Predict
# -------
print('~ predicting ~')
predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis=1)
predictions = pd.Series(predictions, name='Label')
submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
submissions.to_csv(DIR_ARTIFACTS + ARTIFACT_NAME, index=False, header=True)
print('Artifact file generated:')
print(ARTIFACT_NAME)
print('~ done ~')