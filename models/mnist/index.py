import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

dir_data = './data/'
dir_artifacts = './artifacts/'

def printFeatures(dataset):
  for feature_batch in dataset.take(1):
    for key, value in feature_batch.items():
      print("  {!r:20s}: {}".format(key, value))

def validate(data):
  return(data.isnull().any())

def normalize(data):
  normalized = data / 255.0
  return(normalized)

if __name__ == '__main__':

  print('~ training started ~')

  # Load Data
  # ---------
  train = pd.read_csv(dir_data + 'train.csv')
  test = pd.read_csv(dir_data + 'test.csv')

  X_train = train

  # Validate Data
  # -------------
  validate(train)
  validate(test)

  # Train / Test Datasets
  # ---------------------
  X_train = train.iloc[:,1:]
  Y_train = train.iloc[:,0]
  X_test = test

  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.3, random_state=2)

  print('train')
  print(X_train.shape)

  print('val')
  print(X_val.shape)

  print('test')
  print(X_test.shape)

  # Normalize Data
  # --------------
  X_train = normalize(X_train)
  X_val = normalize(X_val)
  X_test = normalize(X_test)


  # Generate Tensors
  # ----------------
  X_train = tf.data.Dataset.from_tensor_slices(dict(X_train))
  X_val = tf.data.Dataset.from_tensor_slices(dict(X_val))
  X_test = tf.data.Dataset.from_tensor_slices(dict(X_test))

  # printFeatures(X_train)

  # Model
  # -----
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  model.fit(X_train, Y_train, epochs=5)
  
  # Validate
  # -------
  print('Validate Performance')
  model.evaluate(X_val, Y_val, verbose=2)

  # Predict
  # -------
  print('Predictions')
  # model.evaluate(X_test, Y_test, verbose=2)
  # predictions = model.predict(X_test)
  # submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
  # submissions.to_csv(dir_artifacts + 'predictions-random-forest.csv', index=False, header=True)