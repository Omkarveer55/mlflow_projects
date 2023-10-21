import logging
import sys
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
#from mlflow.models import infer_signature

def eval(actual,predicted):
    r2score = r2_score(actual,predicted)
    mse = mean_squared_error(actual,predicted)
    mae = mean_absolute_error(actual,predicted)
    return r2score,mse,mae

if __name__ == '__main__':
    df = pd.read_csv('iris.csv')
    X = df.drop('target',axis=1)
    y = df['target']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)
    model = ElasticNet()
    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    with mlflow.start_run():
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        r2score,mse,mae = eval(y_test,y_pred)
        mlflow.log_metric("r2_score",r2score)
        mlflow.log_metric("mse",mse)
        mlflow.log_metric("mae",mae)
        mlflow.log_param("alpha",alpha)
        mlflow.log_param("l1_ratio",l1_ratio)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
        else:
            mlflow.sklearn.log_model(model, "model")