# Model train python script -- Extract code from model training notebook

from sklearn.model_selection import train_test_split
import numpy as np 

def train_model(train_data):
  corr = train_data.corr(numeric_only = True)
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

  return x_train, y_train, x_test, y_test