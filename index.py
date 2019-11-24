import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# random forest
# ensemble
# CNNs

def validate(data):
  return(data.isnull().any())

def normalize(data):
  normalized = data / 255.0
  return(normalized)

def renderDigitImage(digit, pixels, title):
  plt.subplot(330 + (i+1))
  plt.imshow(pixels, cmap=plt.get_cmap('gray'))
  plt.title(title)
  plt.show()

if __name__ == "__main__":

  print("~ training started ~")

  # Load Data
  # ---------
  train = pandas.read_csv('data/train.csv')
  test = pandas.read_csv('data/test.csv')

  # Validate Data
  # -------------
  validate(train)
  validate(test)

  # Visually Inspect Data
  # ---------------------
  # X_train = X_train.reshape(X_train.shape[0], 28, 28)
  # for i in range(6, 9):
  #   renderDigitImage(i, X_train[i], y_train[i])

  # Train / Test Datasets
  # ---------------------
  X_train = (train.iloc[:,1:].values).astype('float32') 
  Y_train = train.iloc[:,0].values.astype('int32')
  X_test = test.values.astype('float32')

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

  # Model
  # -----
  clf=RandomForestClassifier(n_estimators=100)
  clf.fit(X_train,Y_train)

  # Predict
  # -------
  Y_pred=clf.predict(X_val)

  # Accuracy
  # --------
  print("Accuracy:", metrics.accuracy_score(Y_val, Y_pred))