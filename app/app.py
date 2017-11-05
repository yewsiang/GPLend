
from flask import Flask, render_template, url_for, redirect, request

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')
    #return redirect(url_for('managers.home'))

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
	results = [10.2, 20.3]
	
	return render_template('index.html', results=results)
