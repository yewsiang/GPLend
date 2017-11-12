
import os
import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
BASE_DIR = os.path.realpath("..")
sys.path.append(BASE_DIR)

from data import load_dataset, get_train_test_split


# Load data
filename = os.path.join(BASE_DIR, "dataset", "LoanStats3a.csv")
features, data = load_dataset(filename)
print("Data shape: %s" % str(features.shape))

X_train, X_test, y_train, y_test = get_train_test_split(features, test_size=0.3, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Temporarily use subset of data to debug faster
# TODO: Remove
X_train, y_train = X_train[:1000,:], y_train[:1000]
X_val, y_val     = X_val[:500,:], y_val[:500]
X_test, y_test   = X_test[:500,:], y_test[:500]

# Normalize
X_scaler = MinMaxScaler()
X_scaler.fit(X_train)
X_train = X_scaler.transform(X_train)
X_val = X_scaler.transform(X_val)
X_test = X_scaler.transform(X_test)

print("X_train: %s, y_train: %s" % (str(X_train.shape), str(y_train.shape)))
print("X_val: %s, y_val: %s" % (str(X_val.shape), str(y_val.shape)))
print("X_test: %s, y_test: %s" % (str(X_test.shape), str(y_test.shape)))


# Gaussian Process
import GPy
from sklearn.svm import SVR
from simulation import simulate_N_time_periods

# Normalize
y_scaler = MinMaxScaler()
y_scaler.fit(y_train.reshape(-1,1))
y_train_scaled = y_scaler.transform(y_train.reshape(-1,1))

# Initialize GP Model
kernel = GPy.kern.RBF(input_dim=X_train.shape[1], variance=1., lengthscale=1.)
gp_model = GPy.models.GPRegression(X_train, y_train_scaled, kernel)
gp_model.optimize()

SEED            = 1
THRESHOLD       = 1.1
NUM_PERIODS     = 2#5
NUM_MONTHS      = 5#30
FUND_GIVEN      = 1e6
LOANS_PER_MONTH = 10#100
CONF_QUANTILE   = (40,100)

perf_gp = simulate_N_time_periods(gp_model, X_val, y_val, X_scaler, y_scaler, 
                                  threshold=THRESHOLD, num_periods=NUM_PERIODS, fund_given=FUND_GIVEN, 
                                  num_months=NUM_MONTHS,incoming_loans_per_time_period=LOANS_PER_MONTH,
                                  conf_quantile=CONF_QUANTILE, optimize_for="TODO", 
                                  version="loan_amount_and_variance", model_type="gp", seed=SEED)

svm_rbf = SVR(kernel="rbf", C=1e4)
svm_rbf.fit(X_train, y_train)
perf_others = simulate_N_time_periods(svm_rbf, X_val, y_val, X_scaler, y_scaler, 
                                      threshold=THRESHOLD, num_periods=NUM_PERIODS, fund_given=FUND_GIVEN, 
                                      num_months=NUM_MONTHS, incoming_loans_per_time_period=LOANS_PER_MONTH,
                                      conf_quantile=None, optimize_for="TODO", 
                                      version="loan_amount", model_type="others", seed=SEED)

print("Mean Total Profits:")
print(np.mean(np.sum(perf_gp[:,:,0], axis=1)))
print(np.mean(np.sum(perf_others[:,:,0], axis=1)))


# Web app
from flask import Flask, render_template, url_for, redirect, request
app = Flask(__name__)


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/borrowers')
def borrowers():
    return render_template('borrowers.html')

@app.route('/managers')
def managers():
    return render_template('index.html')

@app.route('/run', methods=['GET'])
def run():
	params               = request.args
	active_learning_algo = params['active_learning']
	bayesian_opt_algo    = params['bayes_opt']
	n_sims               = params['n_sims']
	n_months             = params['n_months']
	n_loans_per_month    = params['n_loans_per_month']
	initial_funds        = params['initial_funds']
	confidence           = params['confidence']

	# Set default values
	if n_sims == '':
		n_sims = 20
	if n_months == '':
		n_months = 60
	if n_loans_per_month == '':
		n_loans_per_month = 20
	if initial_funds == '':
		initial_funds = 1000000
	if confidence == '':
		confidence = 0.7

	print('=== USER SPECIFIED PARAMETERS ===')
	print("\n[Parameters]")
	print(params)

	print("\n[Algorithms]")
	print("Active Learning   : %s" % active_learning_algo)
	print("Bayesian Opt      : %s" % bayesian_opt_algo)

	print("\n[Simulation Parameters]")
	print("N_sims            : %s" % n_sims)
	print("N_months          : %s" % n_months)
	print("N_loans_per_month : %s" % n_loans_per_month)
	print("Initial Funds     : %s" % initial_funds)
	print("Confidence        : %s\n" % confidence)

	# Compute simulation
	import mpld3
	from visualisation import plot_portfolio_performance_comparisons

	fig = plot_portfolio_performance_comparisons([perf_gp, perf_others], 
		figsize=(16,8), return_figure=True, legend_names=["GP", "Others"])
	results = mpld3.fig_to_html(fig)
	
	return render_template('index.html', results=results)
