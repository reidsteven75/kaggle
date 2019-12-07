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
DIR_ARTIFACTS = './artifacts'
DIR_CHECKPOINT = './artifacts/cancer-detector.ckpt'

ARTIFACT_NAME = 'predictions.csv'

VAL_TRAIN_RATIO = 0.3  # VAL / TEST

IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)
LEARNING_RATE = 0.0001 / 10
TRAINING_EPOCHS = 30

CLASS_ENCODINGS = {
  'none': 0,
  'tumor': 1 
}

def _encodeImage(filepath, label):
  img = tf.io.read_file(filepath)
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

def generateDataFrame(label, images):
  df = pd.DataFrame(list(zip([label]*len(images), images)), columns=[TARGET, 'image'])
  return(df)

def parseFolders(filepaths_dict):
  Y = []
  X = []
  for label, filepaths in filepaths_dict.items():
    for filepath in filepaths: 
      X.append(filepath)
      Y.append(CLASS_ENCODINGS[label])
  dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(X), tf.constant(Y))
  )
  return(dataset)
  
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
dev = parseFolders(dev_filepaths)
dev = (dev.map(_encodeImage)
      .shuffle(buffer_size=10000)
      .batch(BATCH_SIZE)
      )

train = parseFolders(train_filepaths)
train = (train.map(_encodeImage)
      .shuffle(buffer_size=10000)
      .batch(BATCH_SIZE)
      )

test = parseFolders(test_filepaths)
test = (test.map(_encodeImage)
      .shuffle(buffer_size=10000)
      .batch(BATCH_SIZE)
      )

print('~ transfer learning from MobileNetV2 ~')
base_model = tf.keras.applications.MobileNetV2(
  input_shape=IMG_SHAPE,
  include_top=False,
  weights='imagenet'
)
base_model.trainable = False

print('~ configuring & compiling model ~')
model = tf.keras.Sequential([
  base_model,
  tf.keras.layers.GlobalMaxPooling2D(),
  tf.keras.layers.Dense(1, activation='sigmoid')
  # tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5),padding='Same', activation='relu', input_shape=(IMAGE_SIZE,IMAGE_SIZE,3)),
  # tf.keras.layers.Conv2D(filters=32, kernel_size=(5,5),padding='Same', activation='relu'),
  # tf.keras.layers.MaxPool2D(pool_size=(2,2)),
  # tf.keras.layers.Dropout(0.25),
  # tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'),
  # tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),padding='Same', activation='relu'),
  # tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
  # tf.keras.layers.Dropout(0.25),
  # tf.keras.layers.Flatten(),
  # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  # tf.keras.layers.Dense(128, activation='relu'),
  # tf.keras.layers.Dropout(0.2),
  # tf.keras.layers.Dense(1, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), 
              loss='binary_crossentropy',
              metrics=['accuracy'])

print('~ training model ~')
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=DIR_CHECKPOINT,
                                                 save_weights_only=True,
                                                 verbose=1)

model.fit(train, 
          epochs=TRAINING_EPOCHS,
          callbacks=[cp_callback])

print('~ scoring model ~')
model.evaluate(test, verbose=2)
exit()

# # Predict
# # -------
# print('~ predicting ~')
# predictions = model.predict(X_test)
# predictions = np.argmax(predictions,axis=1)
# predictions = pd.Series(predictions, name='Label')
# submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
# submissions.to_csv(DIR_ARTIFACTS + ARTIFACT_NAME, index=False, header=True)
# print('Artifact file generated:')
# print(ARTIFACT_NAME)
# print('~ done ~')