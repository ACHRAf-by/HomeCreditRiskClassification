import mlflow
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from predict_model import *

def run_mlflow(x_train, y_train, x_test, y_test):
  with mlflow.start_run() as run:
    model = xgb.XGBClassifier(seed = 9999,
                              n_jobs = -10,
                              base_score = 0.5,
                              booster = 'gbtree',
                              gamma = 0.3,
                              learning_rate = 0.1,
                              reg_alpha = 1,
                              reg_lambda = 0.5,
                              eval_metric = 'mlogloss')

    model.fit(x_train, y_train)

    y_pred = predict_model(model, x_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy: {:.2f}%".format(acc*100))
    print("Precision: {:.2f}%".format(prec*100))
    print("Recall: {:.2f}%".format(rec*100))
    print("F1 Score: {:.2f}".format(f1))

    mlflow.xgboost.log_model(model, "model")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    run_id = run.info.run_id

  return run_id