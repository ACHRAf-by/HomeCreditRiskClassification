import mlflow 
import mlflow.xgboost as mxgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np 
import pandas as pd

def train_model(train_data):
  corr = train_data.corr()
  high_corr = corr[((corr > 0.05) | (corr < -0.05)) & (corr < 1) ]

  correlated_columns = {}
  for col in high_corr.columns:
    correlated_features = high_corr.columns[(~high_corr[col].isna())].tolist()
    correlated_features = list(set(correlated_features).difference(set(correlated_columns.keys())))
    correlated_columns[col] = correlated_features
  
  selected_features = correlated_columns["TARGET"]
  for feature in selected_features.copy():
    correlated_correlated_features = high_corr[feature].abs().sort_values(ascending=False)

    features_to_select = correlated_correlated_features[correlated_correlated_features < 90][:3].index.tolist()
    selected_features.extend(features_to_select)
    
  x = train_data.loc[:, np.unique(selected_features)].drop(columns="TARGET")
  y = train_data.loc[:, "TARGET"]

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

  with mlflow.start_run():
    model = xgb.XGBClassifier(seed = 9999,
                              n_jobs=-10,
                              base_score=0.5,
                              booster= 'gbtree',
                              gamma= 0.3,
                              learning_rate= 0.1,
                              reg_alpha= 1,
                              reg_lambda= 0.5,
                              eval_metric='mlogloss')

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)

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

  return model,x

train = pd.read_csv('../models/csv/application_train.csv')

train_model(train)
