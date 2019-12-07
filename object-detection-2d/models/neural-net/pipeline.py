#!/usr/bin/python
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate

import utils
import os

DIR_DATA = './data/brain_tumors'
DIR_ARTIFACTS = './artifacts'
ARTIFACT_NAME = 'predictions.csv'
TRAINING_EPOCHS = 15
VAL_TRAIN_RATIO = 0.3  # VAL / TEST

IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2

def parse_image(file_path):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img)
  img = (tf.cast(img, tf.float32)/127.5) - 1
  img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  return(img)

def getFilePaths(dir):
  files = []
  for file in os.listdir(dir):
    files.append(dir + '/' + file)
  return(files)

# Load Data
# ---------
print('~ loading data ~')
train_file_paths = {
  'none': getFilePaths(DIR_DATA + '/train/none'),
  'tumor': getFilePaths(DIR_DATA + '/train/tumor')
}
test_file_paths = {
  'none': getFilePaths(DIR_DATA + '/test/none'),
  'tumor': getFilePaths(DIR_DATA + '/test/tumor')
}

train_file_paths['none']
train_file_paths['tumor']

test_file_paths['none']
test_file_paths['tumor']

image = parse_image(train_file_paths['none'][0])

train_images_none = [parse_image(item) for item in train_file_paths['none']]
train_images_tumor = [parse_image(item) for item in train_file_paths['tumor']]

test_images_none = [parse_image(item) for item in test_file_paths['none']]
test_images_tumor = [parse_image(item) for item in test_file_paths['tumor']]

exit()

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