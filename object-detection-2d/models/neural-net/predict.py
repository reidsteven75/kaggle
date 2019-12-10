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
DIR_CHECKPOINT = './checkpoints'

IMAGE_SIZE = 96 # Minimum image size for use with MobileNetV2
BATCH_SIZE = 32
IMG_SHAPE = (IMAGE_SIZE, IMAGE_SIZE, 3)

CLASS_ENCODINGS = {
  'none': 0,
  'tumor': 1 
}

def _encodeImage(filepath):
  img = tf.io.read_file(filepath[0])
  img = tf.image.decode_png(img, channels=3) # RGB
  img = (tf.cast(img, tf.float32)/127.5) - 1
  img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
  print(img)
  return(img)

def getFilePaths(dir):
  files = []
  for file in os.listdir(dir):
    if file.endswith('.png'):
      files.append(dir + '/' + file)
  return(files)

print('~ load data ~')
dataset = getFilePaths(DIR_DATA + '/predict')
print(dataset)
dataset = tf.data.Dataset.from_tensor_slices(
    (tf.constant(dataset))
  )
print(dataset)
dataset = (dataset.map(_encodeImage))

print(dataset)

print('~ load model ~')
model = tf.keras.models.load_model(DIR_CHECKPOINT + '/cancer-detector-2d.h5')

print('~ predicting ~')
# print(test_df['image'])
# results = test_df['image']
predictions = model.predict(dataset)
print(predictions)
# results['label_prediction'] = predictions[0]
# print(results)


# submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), TARGET: predictions})
# submissions.to_csv(DIR_ARTIFACTS + ARTIFACT_NAME, index=False, header=True)
# print('Artifact file generated:')
# print(ARTIFACT_NAME)
print('~ done ~')