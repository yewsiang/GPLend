from kernels import get_optimized_model, get_predictions
from optimization import gp_optimize_threshold
from simulation import lend_using_target_only


def partition_by_relevance(test_point, top_size, X_train, kernel):
    """
    Given a test data point, finds the most k relevant (similar) data points in the training set,
    which, when combined with the given data point, gives the top k kernel value
    :param test_point: test data point
    :param X_train: training set
    :param kernel: kernel of choice
    :param top_size: number of top values
    :return (Top k relevant points, Rest)
    """
    # Covariance matrix K(X_train, test_point)
    cov = kernel.K(X_train, test_point.reshape(1, -1))

    # Partition into top k and the rest
    partitions = cov.reshape(-1).argpartition(top_size)
    top_ind = partitions[-top_size:]
    rest_ind = partitions[:-top_size]

    return top_ind, rest_ind


def test_kernel_relevance_single(kernel, X_train, y_train, X_val, y_val,
                                 x_test, y_test, X_scaler, y_scaler,
                                 top_size, optimize_for='profits'):

    _, i_irre = partition_by_relevance(x_test, top_size, X_train, kernel)
    X_irre = X_train[i_irre]
    y_irre = y_train[i_irre]
    gp_model = get_optimized_model(X_irre, y_irre, kernel)
    print("Mean: %s" % gp_model.mean_function)
    print("Kernel: %s" % gp_model.kern)

    # Optimize threshold for profits / profit_percentage
    threshold = gp_optimize_threshold(gp_model, X_val, y_val,
                                      X_scaler, y_scaler, optimize_for=optimize_for)

    # Test threshold value on test set
    regressed_payment, var = gp_model.


def test_kernel_relevance(kernel, X_train, y_train, X_val, y_val,
                          X_test, y_test, X_scaler, y_scaler, optimize_for='profits'):
    gp_model = get_optimized_model(X_train, y_train, kernel)
    print("Mean: %s" % gp_model.mean_function)
    print("Kernel: %s" % gp_model.kern)

    # Use X_val (validation set) to optimize threshold

    # Optimize threshold for profits / profit_percentage
    threshold = gp_optimize_threshold(gp_model, X_val, y_val,
                                      X_scaler, y_scaler, optimize_for=optimize_for)

    # Test threshold value on test set
    regressed_payment, var = get_predictions(gp_model, X_test, y_scaler=y_scaler)
    print("\n============== Kernel: {} ====================".format(kernel.name))
    print("\n----------- Testing on X_test ------------")
    lend_using_target_only(regressed_payment, X_test, y_test, X_scaler, threshold=1.0)
    lend_using_target_only(regressed_payment, X_test, y_test, X_scaler, threshold=threshold)

    print(threshold)