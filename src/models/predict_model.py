def predict_model(model, x_test):
  y_pred = model.predict(x_test)
  print(f"Predictions:\n{y_pred}")
  return y_pred
