import GPy
import numpy as np

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
    X_u = X
    y_u = y

    # Initialize GP Model
    kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)
    
    for i in range(sampleSize - 1):
        # update gp model
        gp_model = GPy.models.GPRegression(X_o, y_o, kernel)
        
        # remove observed points
        np.delete(X_u, selected_index, axis=0)
        np.delete(y_u, selected_index)
        
        # find point with max posterior variance and add it to the collection of observed points
        _, conf = gp_model.predict(X_u)
        selected_index = np.argmax(conf)
        np.append(X_o, X_u[selected_index, :])
        np.append(X_o, y_u[selected_index])

    return X_o, y_o

    
def random_sampling(X, y, percentage):
    size, dim = X.shape
    sampleSize = int(size * percentage)
    selected_index = np.random.choice(size, sampleSize, replace=False)
    X_o = X[selected_index, :]
    y_o = y[selected_index]
    return X_o, y_o

