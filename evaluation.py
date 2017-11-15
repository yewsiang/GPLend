import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, HuberRegressor
from simulation import lend_using_target_only

def evaluate_model(model, X, y):
  pred = model.predict(X)
  score = model.score(X, y)
  mean_abs_err = np.mean(np.abs(pred - y))

  print("Mean absolute error: %.3f" % mean_abs_err)
  print("R^2 Score:           %.3f" % score)

def evaluate_gp_model(model, X, y, y_scaler):
  pred, conf = model.predict(X)
  pred = y_scaler.inverse_transform(pred)
  mean_abs_err = np.mean(np.abs(pred - y))

  print("Mean absolute error: %.3f" % mean_abs_err)

def train_and_test_other_models(X_train, y_train, X_test, y_test, X_scaler):
  # Linear Regression
  print("\n-- Linear Regression --")
  lin_reg = LinearRegression()
  lin_reg.fit(X_train, y_train)
  evaluate_model(lin_reg, X_test, y_test)

  # Huber Regressor
  print("\n-- Huber Regressor --")
  hub_reg = HuberRegressor(epsilon=1.)
  hub_reg.fit(X_train, y_train)
  evaluate_model(hub_reg, X_test, y_test)

  # Linear SVM
  print("\n-- Linear SVM --")
  svm_lin = SVR(kernel="linear", C=1e3)
  svm_lin.fit(X_train, y_train)
  evaluate_model(svm_lin, X_test, y_test)

  # Poly SVM
  print("\n-- Poly SVM 2 --")
  svm_poly_2 = SVR(kernel="poly", degree=2, C=1e5)
  svm_poly_2.fit(X_train, y_train)
  evaluate_model(svm_poly_2, X_test, y_test)

  print("\n-- Poly SVM 5 --")
  svm_poly_5 = SVR(kernel="poly", degree=5, C=1e8)
  svm_poly_5.fit(X_train, y_train)
  evaluate_model(svm_poly_5, X_test, y_test)

  # RBF SVM
  print("\n-- RBF SVM --")
  svm_rbf = SVR(kernel="rbf", C=1e4)
  svm_rbf.fit(X_train, y_train)
  evaluate_model(svm_rbf, X_test, y_test)

  # Simulate making loans only to those who can pay back threshold times of original amount
  regressed_payment = lin_reg.predict(X_test)
  lend_using_target_only(regressed_payment, X_test, y_test, X_scaler, threshold=1.0)
  