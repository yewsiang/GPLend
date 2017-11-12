import GPy
import math
import numpy as np
from features import *

# X, y are dataset input and output
# percentage is the percentage of samples to be picked from the dataset
# function returns the collection of selected samples from the dataset

def maximum_entropy_sampling(X, y, percentage):
    # pick two points to initialize a GP
    selected_index = [0,1]
    X_o = X[selected_index,:]
    y_o = y[selected_index]
    size,dim = X.shape
    sampleSize = int(size * percentage)
    X_u = np.copy(X)
    y_u = np.copy(y)

    # Initialize GP Model
    kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)
    
    for i in range(sampleSize - 2):
        # update gp model
        gp_model = GPy.models.GPRegression(X_o, y_o, kernel)
        gp_model.optimize()

        # remove observed points
        X_u = np.delete(X_u, selected_index, axis=0)
        y_u = np.delete(y_u, selected_index)
        
        # find point with max posterior variance and add it to the collection of observed points
        _, conf = gp_model.predict(X_u)
        selected_index = np.argmax(conf)
        X_o = np.vstack((X_o, X_u[selected_index, :]))
        y_o = np.vstack((y_o, y_u[selected_index]))

    return X_o, y_o

def maximum_mutual_information_sampling(X, y, percentage):
    selected_index = [0,1]
    X_o = X[selected_index,:]
    y_o = y[selected_index]
    size,dim = X.shape
    sampleSize = int(size * percentage)
    X_u = np.copy(X)
    y_u = np.copy(y)

    kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)

    for i in range(sampleSize - 2):
        gp_model_o = GPy.models.GPRegression(X_o, y_o, kernel)
        gp_model_o.optimize()

        # remove observed points
        X_u = np.delete(X_u, selected_index, axis=0)
        y_u = np.delete(y_u, selected_index)

        max_index = 0
        max_mutual_info = 0

        for j in range(X_u.shape[0]):
            X_selected = X_u[j, :]

            X_u_temp = np.delete(X_u, j, axis=0)
            y_u_temp = np.delete(y_u, selected_index)

            gp_model_u = GPy.models.GPRegression(X_u_temp, y_u_temp, kernel)
            gp_model_u.optimize()

            _, var_o = gp_model_o.predict(X_selected)
            _, var_u = gp_model_u.predict(X_selected)

            mutual_info = var_o // var_u   # 0.5log(var_o) - 0.5log(var_u) = 0.5log(var_o / var_u)
 
            if mutual_info > max_mutual_info:
                max_mutual_info = mutual_info
                max_index = j

        selected_index = j
        X_o = np.vstack((X_o, X_u[selected_index, :]))
        y_o = np.vstack((y_o, y_u[selected_index]))

    return X_o, y_o

def DARE_sampling(X, y, percentage, X_scaler):
    selected_index = [0,1]
    X_o = X[selected_index,:]
    y_o = y[selected_index]
    size,dim = X.shape
    sampleSize = int(size * percentage)
    X_u = np.copy(X)
    y_u = np.copy(y)
    
    # Initialize GP Model
    kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)
    
    for i in range(sampleSize - 2):
        # update gp model
        gp_model = GPy.models.GPRegression(X_o, y_o, kernel)
        gp_model.optimize()

        # remove observed points
        X_u = np.delete(X_u, selected_index, axis=0)
        y_u = np.delete(y_u, selected_index)

        mean, conf = gp_model.predict(X_u)
        etp = get_expected_total_payment(X_u, X_scaler)
        result = np.divide(np.absolute(etp - mean), np.sqrt(conf))
        selected_index = np.argmin(result)
        X_o = np.vstack((X_o, X_u[selected_index, :]))
        y_o = np.vstack((y_o, y_u[selected_index]))

    return X_o, y_o
    
def random_sampling(X, y, percentage):
    size, dim = X.shape
    sampleSize = int(size * percentage)
    selected_index = np.random.choice(size, sampleSize, replace=False)
    X_o = X[selected_index, :]
    y_o = y[selected_index]
    return X_o, y_o

def fixed_sampling(X, y, percentage):
    size, dim = X.shape
    sampleSize = int(size * percentage)
    X_o = X[sampleSize:2*sampleSize, :]
    y_o = y[sampleSize:2*sampleSize]
    return X_o, y_o