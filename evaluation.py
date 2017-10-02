
import numpy as np

def evaluate_model(model, X, y):
  pred = model.predict(X)
  score = model.score(X, y)
  mean_abs_err = np.mean(np.abs(pred - y))

  print("Mean absolute error: %.3f" % mean_abs_err)
  print("R^2 Score:           %.3f" % score)

def print_loan_stats(num_loans, total_loans, loans_given, payments_rec, profits, profit_perc):
  print("Loans approved:    %d/%d" % (num_loans, total_loans))
  print("Loans given:       $ %.1f" % loans_given)
  print("Payments received: $ %.1f\n" % payments_rec)
  print("Profits:           $ %.1f" % profits)
  print("Profit Percentage: %.1f%%" % profit_perc)
    
def lend_using_target_only(regressed_payment, X, y, scaler, threshold=1.0):
  """
  Simulate making loans with the trained model using only the target
  (the regressed total payment of the customer).
  If predicted total payment of customer is below (threshold * X[0]) 
  (where X[0] is the loan amount), reject making the loan.
  """
  loan_amount = scaler.inverse_transform(X)[:,0]
  satisfactory_payment = threshold * loan_amount
  loans_approved = regressed_payment > satisfactory_payment
  
  # Loaning to all
  loans_given_prev = np.sum(loan_amount[:]) / 1000
  payments_prev = np.sum(y) / 1000
  profits_prev = payments_prev - loans_given_prev
  profit_percentage_prev = profits_prev / loans_given_prev * 100
  
  # Loan according to model and threshold
  loans_given = np.sum(loan_amount[loans_approved]) / 1000
  payments = np.sum(y[loans_approved]) / 1000
  profits = payments - loans_given
  profit_percentage = profits / loans_given * 100
  
  print("\n--- Without model ---")
  print_loan_stats(X.shape[0], X.shape[0], loans_given_prev, payments_prev, profits_prev, profit_percentage_prev)
  
  print("\n---- With model ----")
  print_loan_stats(np.sum(loans_approved), X.shape[0], loans_given, payments, profits, profit_percentage)
