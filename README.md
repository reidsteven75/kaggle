Kaggle Competitions
-------------------
Models for various Kaggle competitions

## Prereqs
  - docker
  - docker-compose
  
## Making Predictions

Each competition is implemented with classical machine learning techniques, and some are implemented with more advanced machine learning techniques.

Train & test data can be modified for each model.


`Digit Recognizer`

```
$ docker-compose up --build digit-recognizer-classical
```

```
$ docker-compose up --build digit-recognizer-neural-net
```

`Titanic Survivor Predictor`

```
$ docker-compose up --build titanic-classical
```

`House Price Predictor`

```
$ docker-compose up --build house-prices-classical
```