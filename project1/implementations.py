import numpy as np


#######
# Utils
#######


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


###############
# Preliminaries
###############


def compute_loss_mse(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    e = y - tx @ w
    return ((e.T @ e) / (2 * tx.shape[0])).squeeze()


def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    e = y - tx @ w
    return (-1 / y.shape[0]) * (tx.T @ e)


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        A numpy array of shape (D, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    return compute_gradient_mse(y, tx, w)


def sigmoid(t):
    """Sigmoid function.

    Args:
        t (np.array): Input data of shape (N, )

    Returns:
        np.array: Probabilities of shape (N, ), where each value is in [0, 1].
    """
    return np.exp(-np.logaddexp(0, -t))


def compute_loss_logistic(y, tx, w):
    """Logistic regression loss function for binary classes.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
    Returns:
        scalar: Loss of logistic regression.
    """
    dot_prod = tx @ w
    return np.mean(np.logaddexp(0, dot_prod) - y * dot_prod)


def compute_gradient_logistic(y, tx, w):
    """Logistic regression gradient function for binary classes.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
    Returns:
        numpy array: Gradient array of shape (D, )
    """
    n = tx.shape[0]  # Divide directly to avoid overflow
    return tx.T @ ((sigmoid(tx @ w) / n) - (y / n))


def compute_gradient_reg_logistic(y, tx, w, lambda_):
    """L2 regularized logistic regression gradient function for binary classes.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, ). The vector of model parameters.
        lambda_: scalar, regularization strength
    Returns:
        numpy array: Gradient array of shape (D, )
    """
    return compute_gradient_logistic(y, tx, w) + (2 * lambda_) * w


def logistic_regression_classify(tx, w):
    """Classification function for binary class logistic regression.

    Args:
        tx (np.array): Dataset of shape (N, D).
        w (np.array): Weights of logistic regression model of shape (D, )
    Returns:
        np.array: Label assignments of data of shape (N, )
    """
    predictions = sigmoid(tx @ w)
    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1
    return predictions


############
# Algorithms
############


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma, eps=1e-6):
    """The Gradient Descent (GD) algorithm applied to linear regression with MSE cost function.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        eps: gradient norm breaking threshold, since the loss function is convex

    Returns:
        w: model parameters as a numpy array of shape (D, ) for last iteration of GD
        loss: loss value (scalar) for last iteration of GD
    """
    w = initial_w

    for _ in range(max_iters):
        grad = compute_gradient_mse(y, tx, w)
        w = w - gamma * grad
        if np.linalg.norm(grad) < eps:
            break

    loss = compute_loss_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent (SGD) algorithm applied to linear regression with MSE cost function.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: model parameters as a numpy array of shape (D, ) for last iteration of SGD
        loss: loss value (scalar) for last iteration of SGD
    """
    w = initial_w

    for _ in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 1):
            grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad

    loss = compute_loss_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Calculate the least squares solution.
       Returns MSE, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE loss, scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization strength.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: MSE loss, scalar.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.03947092, 0.00319628])
    """
    w = np.linalg.solve(
        tx.T @ tx + (2 * tx.shape[0] * lambda_) * np.eye(tx.shape[1]), tx.T @ y
    )
    return w, compute_loss_mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma, eps=1e-6):
    """Training function for binary class logistic regression.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        eps: gradient norm breaking threshold, since the loss function is convex

    Returns:
        w: w of shape(D, )
        loss: scalar
    """
    w = initial_w

    for _ in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        if np.linalg.norm(grad) < eps:
            break

    loss = compute_loss_logistic(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, eps=1e-6):
    """Regularized logistic regression training.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization strength.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step size
        eps: gradient norm breaking threshold, since the loss function is convex

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w

    for _ in range(max_iters):
        grad = compute_gradient_reg_logistic(y, tx, w, lambda_)
        w = w - gamma * grad
        if np.linalg.norm(grad) < eps:
            break

    loss = compute_loss_logistic(y, tx, w)
    return w, loss
