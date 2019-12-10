#!/usr/bin/python
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from multiprocessing import Pool
from tqdm import tqdm

import utils
import os

TARGET = 'label'
DIR_DATA = './data/brain_tumors'
DIR_CHECKPOINT = './checkpoints'

ARTIFACT_NAME = 'predictions.csv'

VAL_TRAIN_RATIO = 0.3  # VAL / TEST

IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
LEARNING_RATE = 0.0001 / 10
TRAINING_EPOCHS = 40

CLASS_ENCODINGS = {
  'none': 0,
  'tumor': 1 
}

def _encodeImage(filepath, label):
  img = tf.io.read_file(filepath[0])
  img = tf.image.decode_png(img, channels=3) # RGB
  img = (tf.cast(img, tf.float32)/127.5) - 1
  img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  return(img, label)

def getFilePaths(dir):
  files = []
  for file in os.listdir(dir):
    if file.endswith('.png'):
      files.append(dir + '/' + file)
  return(files)

def generateDataFrame(X, Y):
  df = pd.DataFrame(list(zip(Y, X)), columns=[TARGET, 'image'])
  return(df)

def parseDataFrameToTensor(df, target):
  Y = df.pop(target)
  dataset = tf.data.Dataset.from_tensor_slices(
    (df.values, Y.values)
  )
  return(dataset)

def parseFoldersToDataFrame(filepaths_dict):
  Y = []
  X = []
  for label, filepaths in filepaths_dict.items():
    for filepath in filepaths: 
      X.append(filepath)
      Y.append(CLASS_ENCODINGS[label])
  df = generateDataFrame(X,Y)
  return(df)

def printDatasetX(dataset):
  for x, y in dataset:
    print(x)

def printDatasetY(dataset):
  for x, y in dataset:
    print(y)

print('~ loading data ~')
dev_filepaths = {
  'none': getFilePaths(DIR_DATA + '/dev/none'),
  'tumor': getFilePaths(DIR_DATA + '/dev/tumor')
}
train_filepaths = {
  'none': getFilePaths(DIR_DATA + '/train/none'),
  'tumor': getFilePaths(DIR_DATA + '/train/tumor')
}
test_filepaths = {
  'none': getFilePaths(DIR_DATA + '/test/none'),
  'tumor': getFilePaths(DIR_DATA + '/test/tumor')
}

print('~ prepairing datasets ~')
dev_df = parseFoldersToDataFrame(dev_filepaths)
dev_tensor = parseDataFrameToTensor(dev_df, TARGET)
dev = (dev_tensor.map(_encodeImage)
      .shuffle(buffer_size=10000)
      .batch(BATCH_SIZE)
      )

train_df = parseFoldersToDataFrame(train_filepaths)
train_tensor = parseDataFrameToTensor(train_df, TARGET)
train = (train_tensor.map(_encodeImage)
      .shuffle(buffer_size=10000)
      .batch(BATCH_SIZE)
      )
train_num = train_df.shape[0]

test_df = parseFoldersToDataFrame(test_filepaths)
test_tensor = parseDataFrameToTensor(test_df, TARGET)
test = (test_tensor.map(_encodeImage)
      .shuffle(buffer_size=10000)
      .batch(BATCH_SIZE)
      )

# print('~ transfer learning from MobileNetV2 ~')
# base_model = tf.keras.applications.MobileNetV2(
#   input_shape=IMG_SHAPE,
#   include_top=False,
#   weights='imagenet'
# )
# base_model.trainable = False

print('~ configuring & compiling model ~')
model = tf.keras.Sequential([
  # base_model,
  # tf.keras.layers.GlobalMaxPooling2D(),
  # tf.keras.layers.Dense(1, activation='sigmoid')
  tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
  tf.keras.layers.MaxPool2D(pool_size=(2,2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'),
  tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

print('~ training model ~')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=DIR_CHECKPOINT + '/cancer-detector-2d.ckpt',
                                                 save_weights_only=True,
                                                 verbose=1)

steps_per_epoch = round(train_num/BATCH_SIZE)
print('batch size: ' + str(BATCH_SIZE))
print('# samples: ' + str(train_num))
print('# batches: ' + str(round(train_num/BATCH_SIZE)))
model.fit_generator(train.repeat(), 
          epochs=TRAINING_EPOCHS,
          steps_per_epoch=steps_per_epoch,
          # callbacks=[cp_callback]
          )

print('~ scoring model ~')
model.evaluate_generator(test, verbose=2)

print('~ saving model ~')
model.save(DIR_CHECKPOINT + '/cancer-detector-2d.h5')