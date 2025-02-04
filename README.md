Various predictive models

## Prereqs
  - docker
  - docker-compose
  
## Running Models

Each competition is implemented with classical machine learning techniques, and some are implemented with more advanced machine learning techniques.

Train & test data can be modified for each model.


#### Digit Recognizer

```
$ docker-compose up --build digit-recognizer-classical
```

```
$ docker-compose up --build digit-recognizer-neural-net
```

#### Titanic Survivor Predictor

```
$ docker-compose up --build titanic-classical
```

#### House Price Predictor

```
$ docker-compose up --build house-prices-classical
```

## TODO
  - expose python plotting visualizations
