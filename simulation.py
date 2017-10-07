
import numpy as np
from features import *

def print_loan_stats(num_loans, total_loans, loans_given, payments_rec, profits, profit_perc):
  print("Loans approved:    %d/%d" % (num_loans, total_loans))
  print("Loans given:       $ %.1f" % loans_given)
  print("Payments received: $ %.1f\n" % payments_rec)
  print("Profits:           $ %.1f" % profits)
  print("Profit Percentage: %.1f%%" % profit_perc)
    
def lend_using_target_only(regressed_payment, X, y, X_scaler, threshold=1.0, verbose=True):
  """
  Simulate making loans with the trained model using only the target
  (the regressed total payment of the customer).
  If predicted total payment of customer is below (threshold * X[0]) 
  (where X[0] is the loan amount), reject making the loan.
  """
  loan_amount = get_loan_amnt(X, X_scaler)
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
  
  if verbose:
    #print("\n---- Loan to All ----")
    #print_loan_stats(X.shape[0], X.shape[0], loans_given_prev, payments_prev, profits_prev, profit_percentage_prev)
    
    print("\n---- Threshold: %f ----" % threshold)
    print_loan_stats(np.sum(loans_approved), X.shape[0], loans_given, payments, profits, profit_percentage)

  return profits

def choose_max_loans_given_funds(X_loans_sorted, y_loans_sorted, X_scaler, fund_given):
  """
  Given a list of loans, already sorted by in descending order of how good they are,
  make as many loans as possible greedily without loan amount exceeding FUND_GIVEN.
  """
  if X_loans_sorted.shape[0] == 0:
    # No loans to make
    return np.array([]), np.array([])

  loan_ids = []
  loan_amounts = get_loan_amnt(X_loans_sorted, X_scaler)
  for i in range(loan_amounts.shape[0]):
    loan_amount = loan_amounts[i]
    if fund_given >= loan_amount:
      loan_ids.append(i)
      fund_given -= loan_amount
  return X_loans_sorted[loan_ids,:], y_loans_sorted[loan_ids]

def choose_loans(model, X_loans, y_loans, X_scaler, y_scaler, fund_given, threshold, 
                 optimize_for="profits", version="threshold_only", model_type="gp"):
  """
  Choose the loans to be made using different algorithms and returns the ids.
  Different targets might be optimized for (EG. profits, profit_percentage).
  Fund amount will be considered in all algorithms to make sure that making the loans will not
  exceed the capital of the lender OR for optimization purposes.

  Algorithms:
  1) loan_all - Just lend to everyone without using the mondel
  2) threshold_only - Make a loan using only the threshold value. Sort the loans in descending 
     order of threshold values. Choose the loans with the best threshold values greedily until
     fund_given is no longer able to make any more loans that have exceeded the threshold.
  3) variance
  """
  X_loaned, y_loaned = None, None
  # Evaluate the applicants using the model
  if model_type == "gp":
    y_hat, conf = model.predict(X_loans)
  elif model_type == "others":
    y_hat = model.predict(X_loans)

  # TODO:
  # Optimize_for

  # Different ways of lending
  if version == "loan_all":
    X_loaned, y_loaned = X_loans, y_loans
  
  elif version == "threshold_only":
    regressed_payment = y_scaler.inverse_transform(y_hat).reshape(-1)
    loan_amount = get_loan_amnt(X_loans, X_scaler)

    # Sort loans from the predicted best to the worst
    payment_to_loan_ratio = regressed_payment / loan_amount
    desc_payment_to_loan_ratio_id = np.argsort(-payment_to_loan_ratio)

    desc_payment_to_loan_ratio = payment_to_loan_ratio[desc_payment_to_loan_ratio_id]
    X_loaned = X_loans[desc_payment_to_loan_ratio_id,:]
    y_loaned = y_loans[desc_payment_to_loan_ratio_id]

    # Keep only if it is above threshold
    is_ratio_above_threshold = desc_payment_to_loan_ratio > threshold
    X_loaned = X_loans[is_ratio_above_threshold,:]
    y_loaned = y_loans[is_ratio_above_threshold]
  
  elif version == "variance":
    pass

  # Make sure there is enough money to make these loans
  X_loaned, y_loaned = choose_max_loans_given_funds(X_loaned, y_loaned, X_scaler, fund_given)

  # Sanity check that we have not loaned more money than the fund that we have
  if X_loaned.shape[0] != 0:
    assert(np.sum(get_loan_amnt(X_loaned, X_scaler)) <= fund_given)

  return X_loaned, y_loaned

def simulate_N_time_periods(model, X, y, X_scaler, y_scaler, threshold, num_periods=100, 
                           fund_given=1e7, num_months=180, incoming_loans_per_time_period=50,
                           optimize_for="profits", version="threshold_only", model_type="gp"):
  performances = np.zeros((num_periods, num_months, 4))
  for period in range(num_periods):
    performance = simulate_time_period(model, X, y, X_scaler, y_scaler, threshold,
                                      fund_given=fund_given, 
                                      num_months=num_months, 
                                      incoming_loans_per_time_period=incoming_loans_per_time_period,
                                      optimize_for=optimize_for, 
                                      version=version, 
                                      model_type=model_type)
    performances[period, :] = performance
  return performances

def simulate_time_period(model, X, y, X_scaler, y_scaler, threshold, 
                         fund_given=1e7, num_months=180, incoming_loans_per_time_period=50,
                         optimize_for="profits", version="threshold_only", model_type="gp"):
  """
  Simulate having a portfolio with FUND_GIVEN ($) and NUM_MONTHS (months) to make loans,
  where there will be new INCOMING_LOANS_PER_TIME_PERIOD (loans/month) that is available every month.

  Loans will be made using the model and specified threshold.

  Evaluation metrics such as profits made at the end of the entire time period will be collected.
  """
  N, D = X.shape
  portfolio = Portfolio(fund_given, num_months)

  for t in range(num_months):
    # Simulate interest flows current loans
    portfolio.update_period()

    # Update portfolio
    # TODO

    # Simulate loan applications coming in by sampling from data
    loan_application_ids = np.random.choice(N, incoming_loans_per_time_period, replace=False)
    X_loan_app = X[loan_application_ids, :]
    y_loan_app = y[loan_application_ids] # Actual outcome of loan (which we wouldn't know)

    # Choose the loans to be made using different optim
    X_loaned, y_loaned = choose_loans(model, X_loan_app, y_loan_app, X_scaler, y_scaler, 
                                      portfolio.get_funds(), threshold, 
                                      optimize_for=optimize_for,
                                      version=version,
                                      model_type=model_type)

    # Update portfolio status
    portfolio.make_loans(X_loaned, y_loaned, X_scaler)



    #print("Portfolio funds: %f" % portfolio.get_funds())

  # Report performance
  performance = portfolio.report()
  return performance
    
class Portfolio(object):
  def __init__(self, initial_funds, time_period):
    self.initial_funds = initial_funds
    self.time_period   = time_period
    self.funds         = initial_funds
    self.loans         = []

    # Keeping track of loans across different time periods.
    # EG self.funds_across_time[t] is remaining funds at time t
    self.terms                       = 0
    self.loans_given_across_time     = []
    self.funds_across_time           = []
    self.payments_rec_across_time    = []
    # Profits made the moment the loan was made (used for performance evaluation)
    self.virtual_profits_across_time = [] 

  def make_loans(self, X_loans, y_loans, X_scaler):
    # No loans
    if X_loans.shape[0] == 0:
      self.funds_across_time.append(self.funds)
      self.loans_given_across_time.append(0)
      self.virtual_profits_across_time.append([])
      return

    # Sanity check that we have not loaned more money than the fund than we have
    loan_amt = np.sum(get_loan_amnt(X_loans, X_scaler))
    assert(loan_amt <= self.get_funds())
    self.funds -= loan_amt
    self.funds_across_time.append(self.funds)
    self.loans_given_across_time.append(loan_amt)

    # Keep information about 1) installment, 2) terms remaining, 3) actual payment
    # of every loan
    loan_amts              = get_loan_amnt(X_loans, X_scaler)
    installments_and_etp   = get_installment(X_loans, X_scaler)
    installments           = installments_and_etp[:,0]
    expected_total_payment = installments_and_etp[:,1]
    terms                  = expected_total_payment / installments

    current_virtual_profits = []
    for i in range(X_loans.shape[0]):
      loan_amt      = loan_amts[i]
      installment   = installments[i]
      term          = terms[i]
      total_payment = y_loans[i]
      # Profits the moment the loan has been made (used for performance evaluation)
      current_virtual_profits.append(total_payment - loan_amt)
      self.loans.append([installment, term, total_payment])
    self.virtual_profits_across_time.append(current_virtual_profits)

  def update_period(self):
    updated_loans = []
    payments_rec_per_time_period = []
    for loan in self.loans:
      installment, term, total_payment = loan

      # Add installment payments to the funds
      if total_payment - installment < 0:
        payment_for_period = total_payment
      else:
        payment_for_period = installment
      self.funds += payment_for_period
      payments_rec_per_time_period.append(payment_for_period)

      # Reduce the remaining terms of loans
      term -= 1
      self.terms += 1

      # Reduce the actual payment that was made
      total_payment -= payment_for_period

      if term != 0 and total_payment != 0:
        # Loan is not finished
        updated_loans.append([installment, term, total_payment])

    # Update loans
    self.loans = updated_loans
    self.payments_rec_across_time.append(payments_rec_per_time_period)

  def get_funds(self):
    return self.funds
    
  def report(self):
    # Calculate performance for each time period.
    # Dimension 0: Profits (Total Payment - Loan Amount)
    # Dimension 1: Fund Remaining
    # Dimension 2: Loan amount given in that time period
    # Dimension 3: Payments received in that time period
    performance = np.zeros((self.time_period, 4))
    for t in range(self.time_period):
      performance[t, 0] = np.sum(self.virtual_profits_across_time[t])
      performance[t, 1] = self.funds_across_time[t]
      performance[t, 2] = self.loans_given_across_time[t]
      performance[t, 3] = np.sum(self.payments_rec_across_time[t])
    return performance
