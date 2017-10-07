
# Only these columns will be kept from the raw CSV dataset
# The values that are tagged to these headers are to index them into an array
data_headers = {
    # Inputs
    "loan_amnt":               0, # The listed amount of the loan applied for by the borrower. If at some point in time, the credit department reduces the loan amount, then it will be reflected in this value.
    "term":                    1, # The number of payments on the loan. Values are in months and can be either 36 or 60.
    "int_rate":                2, # Interest Rate on the loan.
    "installment":             3, # The monthly payment owed by the borrower if the loan originates.
    "emp_length":              4, # Employment length in years. Possible values are between 0 and 10 where 0 means less than one year and 10 means ten or more years. 
    "home_ownership":          5, # The home ownership status provided by the borrower during registration or obtained from the credit report. Values are: RENT, OWN, MORTGAGE, OTHER.
    "annual_inc":              6, # The self-reported annual income provided by the borrower during registration.
    "verification_status":     7, # Indicates if income was verified by LC, not verified, or if the income source was verified.
    "purpose":                 8, # A category provided by the borrower for the loan request. 
    "dti":                     9, # Ratio calculated using the borrower’s total monthly debt payments on the total debt obligations, excluding mortgage and the requested LC loan, divided by the borrower’s self-reported monthly income.
    "delinq_2yrs":            10, # The number of 30+ days past-due incidences of delinquency in the borrower's credit file for the past 2 years.
    "inq_last_6mths":         11, # The number of inquiries in past 6 months (excluding auto and mortgage inquiries).
    "mths_since_last_delinq": 12, # The number of months since the borrower's last delinquency.
    "mths_since_last_record": 13, # The number of months since the last public record.
    "open_acc":               14, # The number of months since the last public record.
    "pub_rec":                15, # The number of derogatory public records.
    "total_acc":              16, # The total number of credit lines currently in the borrower's credit file.
    "pub_rec_bankruptcies":   17, # Number of public record bankruptcies.
    
    # Targets
    "loan_status":            18, # Current status of the loan.
    "total_pymnt":            19, # Payments received to date for total amount funded.
}

import csv
from features import *
from preprocessing import *
from collections import OrderedDict
from sklearn.model_selection import train_test_split

# Preprocessing functions for dataset from each of the headers
# Convert the dataset from string to a usable format
preproc_fn = {
    "loan_amnt":              loan_amnt_prefn,
    "term":                   term_prefn,
    "int_rate":               int_rate_prefn,
    "installment":            installment_prefn,
    "emp_length":             emp_length_prefn,
    "home_ownership":         home_ownership_prefn,
    "annual_inc":             annual_inc_prefn,
    "verification_status":    verification_status_prefn,
    "purpose":                purpose_prefn,
    "dti":                    dti_prefn,
    "delinq_2yrs":            delinq_2yrs_prefn,
    "inq_last_6mths":         inq_last_6mths_prefn,
    "mths_since_last_delinq": mths_since_last_delinq_prefn,
    "mths_since_last_record": mths_since_last_record_prefn,
    "open_acc":               open_acc_prefn,
    "pub_rec":                pub_rec_prefn,
    "total_acc":              total_acc_prefn,
    "pub_rec_bankruptcies":   pub_rec_bankruptcies_prefn,
    "loan_status":            loan_status_prefn,
    "total_pymnt":            total_pymnt_prefn
}

# Feature engineering
feature_fn = OrderedDict([
    ("loan_amnt",              loan_amnt_fn),
    ("int_rate",               int_rate_fn),
    ("installment",            installment_fn),
    ("emp_length",             emp_length_fn),
    ("home_ownership",         home_ownership_fn),
    ("annual_inc",             annual_inc_fn),
    ("verification_status",    verification_status_fn),
    ("purpose",                purpose_fn),
    ("dti",                    dti_fn),
    ("delinq_2yrs",            delinq_2yrs_fn),
    ("inq_last_6mths",         inq_last_6mths_fn),
    ("mths_since_last_delinq", mths_since_last_delinq_fn),
    ("mths_since_last_record", mths_since_last_record_fn),
    ("open_acc",               open_acc_fn),
    ("pub_rec",                pub_rec_fn),
    ("total_acc",              total_acc_fn),
    ("pub_rec_bankruptcies",   pub_rec_bankruptcies_fn),
    ("loan_status",            loan_status_fn),
    ("total_pymnt",            total_pymnt_fn)
])

def get_preprocessed_data(filename, cols):
  # Load the relevant attributes and convert them into a usable dtype
  with open(filename, encoding='utf-8') as csvfile:
    data = []
    N_data = len(data_headers)
    reader = csv.reader(csvfile, delimiter=",")
    for i, row in enumerate(reader):
      # Skip first line
      if i == 0: 
        continue
      # Skip abnormal rows
      if len(row) != cols: 
        continue
      # Get headers
      if i == 1: 
        headers = row
        continue
      
      # Get dataset from the remaining rows
      data_line = [0] * N_data
      for j, item in enumerate(row):
        # Ignore features that are not used
        data_attr = headers[j].strip() # Eg loan_amnt, int_rate
        data_attr_id = data_headers.get(data_attr)
        if data_attr_id == None:
          continue
          
        # Convert each string into a usable format
        val = preproc_fn[data_attr](item)
        data_line[data_attr_id] = val
      data.append(data_line)
  return data

def get_features(data):
  # Feature Engineering
  np_data = []
  for row in data:
    np_row = np.array([])

    for _, fn in feature_fn.items():
      feature = fn(row)
      np_row = np.append(np_row, feature)
    np_data.append(np_row)
  np_data = np.vstack(np_data)
  return np_data

def load_dataset(filename):
  cols = 137
  print("Preprocessing...")
  data = get_preprocessed_data(filename, cols)
  print("Feature Engineering...")
  features = get_features(data)
  return features, data

def get_train_test_split(data, test_size=0.3, random_state=0):
  X = data[:,:-2]
  # Note 2nd last col not used for training because it is categorical
  y = data[:,-1]
  X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                      test_size=test_size, 
                                      random_state=random_state)
  return X_train, X_test, y_train, y_test

