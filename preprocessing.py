
# ============= PREPROCESSING FUNCTIONS =============
"""
Simply convert string format of the raw CSV data into a
usable format. Further processing will have to be done 
to convert these values into features for training.
EG. Categorical data into integers
EG. Real values into floats
"""
def loan_amnt_prefn(data_str):
  return int(data_str)

def int_rate_prefn(data_str):
  if data_str[-1] != "%":
    raise Exception("Unexpected data string value: %s" % data_str)
  return float(data_str[:-1])

def installment_prefn(data_str):
  return float(data_str)

def emp_length_prefn(data_str):
  if data_str == "n/a":
    val = 0.
  elif data_str == "< 1 year":
    val = 0.5
  elif data_str == "1 year":
    val = 1
  elif data_str == "10+ years":
    val = 10.
  elif data_str[-5:] == "years":
    val = float(data_str[0])
  else:
    raise Exception("Unexpected data string value: %s" % data_str)
  return val

def home_ownership_prefn(data_str):
  # Categorical
  if data_str == "RENT":
    val = 0
  elif data_str == "MORTGAGE":
    val = 1
  elif data_str == "OWN":
    val = 2
  elif data_str == "NONE":
    val = 3
  elif data_str == "OTHER":
    val = 4
  else:
    raise Exception("Unexpected data string value: %s" % data_str)
  return val

def annual_inc_prefn(data_str):
  if data_str == "":
    return 0
  else:
    return float(data_str)

def verification_status_prefn(data_str):
  # Categorical
  if data_str == "Verified":
    val = 0
  elif data_str == "Source Verified":
    val = 1
  elif data_str == "Not Verified":
    val = 2
  else:
    raise Exception("Unexpected data string value: %s" % data_str)
  return val

def purpose_prefn(data_str):
  # Categorical
  if data_str == "credit_card":
    val = 0
  elif data_str == "car":
    val = 1
  elif data_str == "small_business":
    val = 2
  elif data_str == "wedding":
    val = 3
  elif data_str == "debt_consolidation":
    val = 4
  elif data_str == "small_business":
    val = 5
  elif data_str == "major_purchase":
    val = 6
  elif data_str == "medical":
    val = 7
  elif data_str == "moving":
    val = 8
  elif data_str == "home_improvement":
    val = 9
  elif data_str == "vacation":
    val = 10
  elif data_str == "house":
    val = 11
  elif data_str == "renewable_energy":
    val = 12
  elif data_str == "educational":
    val = 13
  elif data_str == "other":
    val = 14
  else:
    raise Exception("Unexpected data string value: %s" % data_str)
  return val

def dti_prefn(data_str):
  return float(data_str)

def delinq_2yrs_prefn(data_str):
  if data_str == "":
    return 0
  else:
    return int(data_str)

def inq_last_6mths_prefn(data_str):
  if data_str == "":
    return 0
  else:
    return int(data_str)

def mths_since_last_delinq_prefn(data_str):
  if data_str == "":
    return 100000000
  else:
    return int(data_str)

def mths_since_last_record_prefn(data_str):
  if data_str == "":
    return 100000000
  else:
    return int(data_str)

def open_acc_prefn(data_str):
  if data_str == "":
    return 0
  else:
    return int(data_str)

def pub_rec_prefn(data_str):
  if data_str == "":
    return 0
  else:
    return int(data_str)

def total_acc_prefn(data_str):
  if data_str == "":
    return 0
  else:
    return int(data_str)

def pub_rec_bankruptcies_prefn(data_str):
  if data_str == "":
    return 0
  else:
    return int(data_str)

def loan_status_prefn(data_str):
  # Categorical
  if data_str == "Fully Paid":
    val = 0
  elif data_str == "Does not meet the credit policy. Status:Fully Paid":
    val = 0
  elif data_str == "Charged Off":
    val = 1
  elif data_str == "Does not meet the credit policy. Status:Charged Off":
    val = 1
  else:
    raise Exception("Unexpected data string value: %s" % data_str)
  return val

def total_pymnt_prefn(data_str):
  return float(data_str)

