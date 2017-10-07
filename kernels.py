from GPy.models import GPRegression

from optimization import gp_optimize_threshold
from simulation import lend_using_target_only

def get_optimized_model(X_train, y_train, kernel):
    gp_model = GPRegression(X_train, y_train, kernel)
    gp_model.optimize()
    return gp_model


def get_predictions(gp_model, X_test, y_scaler=None):
    y_predict, var = gp_model.predict(X_test)
    if y_scaler != None:
        y_predict = y_scaler.inverse_transform(y_predict).reshape(-1)
    return y_predict, var


def test_kernel(kernel, X_train, y_train, X_val, y_val,
                X_test, y_test, X_scaler, y_scaler, optimize_for='profits'):
    gp_model = get_optimized_model(X_train, y_train, kernel)
    print("Mean: %s" % gp_model.mean_function)
    print("Kernel: %s" % gp_model.kern)

    # Use X_val (validation set) to optimize threshold

    # Optimize threshold for profits / profit_percentage
    threshold = gp_optimize_threshold(gp_model, X_val, y_val, X_scaler, y_scaler, optimize_for=optimize_for)

    # Test threshold value on test set
    regressed_payment, var = get_predictions(gp_model, X_test, y_scaler=y_scaler)
    print("\n============== Kernel: {} ====================".format(kernel.name))
    print("\n----------- Testing on X_test ------------")
    lend_using_target_only(regressed_payment, X_test, y_test, X_scaler, threshold=1.0)
    lend_using_target_only(regressed_payment, X_test, y_test, X_scaler, threshold=threshold)

    print(threshold)
