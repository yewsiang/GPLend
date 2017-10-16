import numpy as np

from GPy.models import GPRegression
from kernels import get_optimized_model, get_predictions
from optimization import gp_optimize_threshold
from simulation import lend_using_target_only


def partition_by_relevance(test_point, top_size, X_train, kernel):
    """
    Given a test data point, finds the most k relevant (similar) data points in the training set,
    which, when combined with the given data point, gives the top k kernel value
    :param test_point: test data point
    :type test_point: np.ndarray (1 x m)
    :param X_train: training set
    :type X_train: np.ndarray (N x m)
    :param kernel: kernel of choice
    :type kernel: GPy.kern
    :param top_size: number of top values
    :return (Top k relevant points, Rest)
    """
    # Covariance matrix K(X_train, test_point)
    cov = -kernel.K(X_train, test_point)

    # Partition into top k and the rest
    partitions = cov.reshape(-1).argpartition(top_size)
    top_ind = partitions[:top_size]
    rest_ind = partitions[top_size:]

    return top_ind, rest_ind


def predict_irrelevant(kernel, X_train, y_train, X_test, y_scaler, top_size):

    _, i_irre = partition_by_relevance(X_test, top_size, X_train, kernel)
    X_irre = X_train[i_irre]
    y_irre = y_train[i_irre]
    gp_model = get_optimized_model(X_irre, y_irre, kernel)

    # Test threshold value on test set
    regressed_payment, var = get_predictions(gp_model, X_test, y_scaler=y_scaler)
    return regressed_payment


def test_kernel_relevance(kernel, X_train, y_train, X_val, y_val,
                          X_test, y_test, X_scaler, y_scaler,
                          top_size, optimize_for='profits'):
    gp_model = get_optimized_model(X_train, y_train, kernel)

    # Optimize threshold for profits / profit_percentage
    threshold = gp_optimize_threshold(gp_model, X_val, y_val,
                                      X_scaler, y_scaler, optimize_for=optimize_for)

    # Test threshold value on test set
    regressed_payment = np.zeros((y_test.size, ))
    print(X_test.shape[0])
    for i in range(X_test.shape[0]):
        regressed_payment[i] = predict_irrelevant(kernel, X_train, y_train,
                                                  X_test[i:i+1, :], y_scaler,top_size)
    print("\n============== Kernel: {} ====================".format(kernel.name))
    print("\n----------- Testing on X_test ------------")
    lend_using_target_only(regressed_payment, X_test, y_test, X_scaler, threshold=1.0)
    lend_using_target_only(regressed_payment, X_test, y_test, X_scaler, threshold=threshold)

    print(threshold)


class GPRelevance:
    X_train = None
    y_train = None
    kernel = None
    top_size = 0

    def __init__(self, X_train, y_train, top_size, kernel=None):
        self.X_train = X_train
        self.y_train = y_train
        self.kernel = kernel

    def predict(self, test_point):
        # Covariance matrix K(X_train, test_point)
        cov = -self.kernel.K(self.X_train, test_point)

        # Partition into top k and the rest
        partitions = cov.reshape(-1).argpartition(self.top_size)
        rest_ind = partitions[self.top_size:]

        X_irre = self.X_train[rest_ind]
        y_irre = self.y_train[rest_ind]
        model = GPRegression(X_irre, y_irre, kernel=self.kernel)
        model.optimize()

        return model.predict(test_point)