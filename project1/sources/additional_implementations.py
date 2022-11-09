from implementations import *


def __backtracking_linesearch(y, tx, w, grad, a_bar=1.0, c=1e-4, rho=0.8):

    a = a_bar
    l_w = compute_loss_logistic(y, tx, w)
    g_w_squared = c * np.linalg.norm(grad)

    while compute_loss_logistic(y, tx, w - a * grad) > l_w - a * g_w_squared:
        a = rho * a
    return a


def reg_logistic_regression_backtracking(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    eps=1e-6,
    a_bar=1.0,
    c=1e-4,
    rho=0.8,
    verbose=False,
    print_period=25,
):
    """Regularized logistic regression training.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization strength.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        eps: gradient norm breaking threshold
        a_bar: maximum step size for backtracking line search
        c: gradient norm reduction factor in backtracking line search
        rho: reduction factor in backtracking line search
        verbose: bool, print information if true
        print_period:

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w

    i = 0
    norm = 0.0
    while i < max_iters:
        grad = compute_gradient_reg_logistic(y, tx, w, lambda_)
        norm = np.linalg.norm(grad)
        if np.linalg.norm(grad) < eps:
            break
        s = __backtracking_linesearch(y, tx, w, grad, a_bar, c, rho)
        w = w - s * grad
        if verbose and i % print_period == 0:
            print(f"Iter {i}: gradient norm {norm}")
        i += 1

    loss = compute_loss_logistic(y, tx, w)
    if verbose:
        print(f"Exited at iter {i} with loss {loss} and norm {norm}.")
    return w, loss


def stochastic_reg_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    gamma,
    batch_size=10,
    eps=1e-6,
    gradient_period=200,
):
    """Stochastic regularized logistic regression training.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization strength.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize
        batch_size: defaults to 10
        eps: gradient norm breaking threshold
        gradient_period: set when to compute full gradient to test if its norm is small enough

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w

    for idx in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient_reg_logistic(minibatch_y, minibatch_tx, w, lambda_)
            w = w - gamma * grad
        if idx % gradient_period == 0:
            if np.linalg.norm(compute_gradient_reg_logistic(y, tx, w, lambda_)) < eps:
                break

    loss = compute_loss_logistic(y, tx, w)
    return w, loss


def stochastic_reg_backtracking_logistic_regression(
    y,
    tx,
    lambda_,
    initial_w,
    max_iters,
    batch_size=10,
    eps=1e-4,
    gradient_period=200,
    verbose=False,
):
    """Stochastic regularized logistic regression training with backtracking linesearch to determine step size.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar, regularization strength.
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        batch_size: defaults to 10
        eps: gradient norm breaking threshold
        gradient_period: set when to compute full gradient to test if its norm is small enough

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w

    idx = 0
    while idx < max_iters:
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient_reg_logistic(minibatch_y, minibatch_tx, w, lambda_)
            s = __backtracking_linesearch(minibatch_y, minibatch_tx, w, lambda_)
            w = w - s * grad
        if idx % gradient_period == 0:
            norm = np.linalg.norm(compute_gradient_reg_logistic(y, tx, w, lambda_))
            if verbose:
                print(f"Iter {idx}, gradient norm: {norm}.")
            if np.linalg.norm(compute_gradient_reg_logistic(y, tx, w, lambda_)) < eps:
                if verbose:
                    print("Gradient small enough")
                break
        idx += 1
    if verbose:
        print(f"Exited at iter {idx}.")
    loss = compute_loss_logistic(y, tx, w)
    return w, loss
