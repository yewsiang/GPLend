
import math
import numpy as np
from data import data_headers

def one_hot_encoding(total_classes, cls):
  if total_classes == 2:
    return cls
  one_hot_vec = np.zeros(total_classes)
  one_hot_vec[cls] = 1
  return one_hot_vec

# ========== FEATURE ENGINEERING FUNCTIONS ==========
"""
Simply convert each row of raw CSV dataset into features
that will be used for training.
Given a dataset row, Each function here will generate a 
scalar/vector that will finally be appended together.
"""
def loan_amnt_fn(row):
  loan_amnt = row[data_headers.get("loan_amnt")]
  return loan_amnt

def int_rate_fn(row):
  int_rate = row[data_headers.get("int_rate")]
  return int_rate

def installment_fn(row):
  installment = row[data_headers.get("installment")]
  term = row[data_headers.get("term")]
  expected_total_payment = installment * term
  return np.array([installment, expected_total_payment])

def emp_length_fn(row):
  emp_length = row[data_headers.get("emp_length")]
  return emp_length

def home_ownership_fn(row):
  home_ownership = row[data_headers.get("home_ownership")]
  one_hot_vec = one_hot_encoding(5, home_ownership)
  return one_hot_vec

def annual_inc_fn(row):
  annual_inc = row[data_headers.get("annual_inc")]
  return annual_inc

def verification_status_fn(row):
  verification_status = row[data_headers.get("verification_status")]
  one_hot_vec = one_hot_encoding(3, verification_status)
  return one_hot_vec

def purpose_fn(row):
  purpose = row[data_headers.get("purpose")]
  one_hot_vec = one_hot_encoding(15, purpose)
  return one_hot_vec

def dti_fn(row):
  dti = row[data_headers.get("dti")]
  return dti

def delinq_2yrs_fn(row):
  delinq_2yrs = row[data_headers.get("delinq_2yrs")]
  return delinq_2yrs

def inq_last_6mths_fn(row):
  inq_last_6mths = row[data_headers.get("inq_last_6mths")]
  return inq_last_6mths

def mths_since_last_delinq_fn(row):
  mths_since_last_delinq = row[data_headers.get("mths_since_last_delinq")]
  return math.log(mths_since_last_delinq + 1e-3)

def mths_since_last_record_fn(row):
  mths_since_last_record = row[data_headers.get("mths_since_last_record")]
  return math.log(mths_since_last_record + 1e-3)

def open_acc_fn(row):
  open_acc = row[data_headers.get("open_acc")]
  return open_acc

def pub_rec_fn(row):
  pub_rec = row[data_headers.get("pub_rec")]
  return pub_rec

def total_acc_fn(row):
  total_acc = row[data_headers.get("total_acc")]
  return total_acc

def pub_rec_bankruptcies_fn(row):
  pub_rec = row[data_headers.get("pub_rec")]
  return pub_rec

def loan_status_fn(row):
  loan_status = row[data_headers.get("loan_status")]
  one_hot_vec = one_hot_encoding(2, loan_status)
  return one_hot_vec

def total_pymnt_fn(row):
  total_pymnt = row[data_headers.get("total_pymnt")]
  #loan_amnt = row[data_headers.get("loan_amnt")]
  #percentage_of_loan_returned = total_pymnt / loan_amnt
  return total_pymnt #percentage_of_loan_returned