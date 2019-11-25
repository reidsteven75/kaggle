import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
# from keras.utils.np_utils import to_categorical

DIR_DATA = './data/'
DIR_ARTIFACTS = './artifacts/'

TRAINING_EPOCHS = 1
VAL_TRAIN_RATIO = 0.3  # VAL / TEST

def validate(data):
  return(data.isnull().any())

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
  validate(train)
  validate(test)

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
  
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = VAL_TRAIN_RATIO, random_state=2)

  print('train x')
  print(X_train.shape)

  print('train y')
  print(Y_val.shape)

  print('val x')
  print(X_val.shape)

  print('val y')
  print(Y_val.shape)

  print('test x')
  print(X_test.shape)

  print('num classes')
  print(Y_train.shape[1])

  # dataset_train = tf.data.Dataset.from_tensor_slices((X_train.values, Y_train.values))
  # dataset_train = dataset_train.shuffle(len(X_train)).batch(1)
  
  # dataset_val = tf.data.Dataset.from_tensor_slices((X_val.values, Y_val.values))
  # dataset_val = dataset_val.shuffle(len(X_train)).batch(1)

  # dataset_test_x = np.array(X_test.values)

  # dataset_test_x = tf.data.Dataset.from_tensor_slices((X_test.values))
  # dataset_test_x = dataset_test_x.shuffle(len(X_test)).batch(1)

  # Model
  # -----
  print('~ compiling model ~')
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])
  print("input shape ",model.input_shape)
  print("output shape ",model.output_shape)

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  print('~ training model ~')
  model.fit(np.array(X_train), np.array(Y_train), epochs=TRAINING_EPOCHS)
  
  # Validate
  # -------
  print('~ validating model performance ~')
  model.evaluate(np.array(X_val), np.array(Y_val), verbose=2)

  # Predict
  # -------
  print('~ predicting ~')
  predictions = model.predict(dataset_test_x)
  print(predictions)
  predictions = np.array(predictions).flatten()
  submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
  submissions.to_csv(DIR_ARTIFACTS + 'predictions-mnist.csv', index=False, header=True)