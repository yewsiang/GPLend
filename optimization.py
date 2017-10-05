
import numpy as np

def gp_optimize_threshold(gp_model, X_val, y_val, X_scaler, y_scaler, optimize_for="profits"):
  """
  Optimize threshold value for a specified target (profits, profit_percentage) 
  on validation set.
  """ 
  y_hat, conf = gp_model.predict(X_val)
  regressed_payment = y_scaler.inverse_transform(y_hat).reshape(-1)
  loan_amt = X_scaler.inverse_transform(X_val)[:,0]

  # This ratio is a guage of how likely a person will pay back.
  # It is compared with a threshold to determine whether or not to loan.
  payment_to_loan_ratio = regressed_payment / loan_amt

  # Sort in descending order
  sorted_ind = np.argsort(-payment_to_loan_ratio)
  sorted_payment_to_loan_ratio = payment_to_loan_ratio[sorted_ind]
  X_sorted, y_sorted = X_val[sorted_ind,:], y_val[sorted_ind]

  threshold, highest_opt_val = 0, 0
  for i, thresh in enumerate(sorted_payment_to_loan_ratio):    
    X_loanee = X_sorted[:i+1,:]
    y_loanee = y_sorted[:i+1]
    
    loan_amt_loanee = np.sum(X_scaler.inverse_transform(X_loanee)[:,0])
    payments_loanee = np.sum(y_loanee)

    # Optimize for different values
    if optimize_for == "profits":
      opt_val = payments_loanee - loan_amt_loanee
    elif optimize_for == "profit_percentage":
      opt_val = (payments_loanee - loan_amt_loanee) / loan_amt_loanee
    else:
      raise Exception("Illegal optimize_for value: %s" % optimize_for)

    # Keep track of highest value (that is being optimized for)
    if opt_val > highest_opt_val:
      threshold = thresh
      highest_opt_val = opt_val
  return threshold