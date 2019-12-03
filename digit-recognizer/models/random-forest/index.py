import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

dir_data = './data/'
dir_artifacts = './artifacts/'

# random forest
# CNNs (pytorch vs. tensorflow)

def validate(data):
  return(data.isnull().any().any())

def normalize(data):
  normalized = data / 255.0
  return(normalized)

if __name__ == '__main__':

  print('~ training started ~')

  # Load Data
  # ---------
  train = pd.read_csv(dir_data + 'train.csv')
  test = pd.read_csv(dir_data + 'test.csv')

  # Validate Data
  # -------------
  print('~ validating data ~')
  if (validate(train) == True or validate(test) == True):
    print('dataset(s) not formatted correctly')
    quit()

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
  model = RandomForestClassifier(n_estimators=100)
  model.fit(X_train,Y_train)

  # Validate
  # -------
  Y_pred=model.predict(X_val)
  print('Accuracy:', metrics.accuracy_score(Y_val, Y_pred))

  # Predict
  # -------
  predictions = model.predict(X_test)
  submissions=pd.DataFrame({'ImageId': list(range(1,len(predictions)+1)), 'Label': predictions})
  submissions.to_csv(dir_artifacts + 'predictions-random-forest.csv', index=False, header=True)