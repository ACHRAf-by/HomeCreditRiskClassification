
import pandas as pd
from train_model import *
from run_mlflow import *

train_data = pd.read_csv('../data/application_train.csv')

x_train, y_train, x_test, y_test = train_model(train_data)

run_id = run_mlflow(x_train, y_train, x_test, y_test)

client = mlflow.tracking.MlflowClient()
metrics_accuracy = client.get_metric_history(run_id, 'accuracy')
metrics_precision = client.get_metric_history(run_id, 'precision')
metrics_recall = client.get_metric_history(run_id, 'recall')
metrics_f1_score = client.get_metric_history(run_id, 'f1_score')

print(f"Run ID: {run_id}\n"
      + f"Accuracy: {metrics_accuracy}\n"
      + f"Precision: {metrics_precision}\n"
      + f"Recall: {metrics_recall}\n"
      + f"F1 score: {metrics_f1_score}\n")