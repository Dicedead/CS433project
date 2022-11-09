import implementations as imp

from additional_implementations import *


def k_fold_cross_validation_reg_log(
    y, tx, k_fold, lambdas, gammas, seed=12, initial_w=None, max_iters=3000
):
    """Cross validation over regularisation parameters lambda and gamma of regularized logistic regression.

    Args:
        y:         shape=(N,)
        tx:        shape=(N,D)
        k_fold:    integer, the number of folds
        lambdas:   shape = (p, ) where p is the number of values of lambda to test
        gammas:    shape = (l, ) where l is the number of values of gamma to test
        seed:      randomness replication
        initial_w: w at iteration 0 in gradient descent
        max_iters: max iterations of gradient descent
    Returns:
        best_lambda : scalar, value of the best lambda
        best_gamma : scalar, value of the best gamma
        best_loss : scalar, the associated root mean squared error for the best lambda
    """

    k_indices = __build_k_indices(y, k_fold, seed)
    best_lambda_index = 0
    best_gamma_index = 0
    best_loss = np.inf

    for i, lambda_ in enumerate(lambdas):
        for j, gamma in enumerate(gammas):
            print("==================")
            print(
                f"Starting validation for lambda={lambda_} ({i}) and gamma={gamma} ({j})."
            )
            loss = 0.0
            for k in range(k_fold):
                loss += __cross_validation_one_iter_reg_log(
                    y,
                    tx,
                    k_indices,
                    k,
                    lambda_,
                    gamma,
                    initial_w=initial_w,
                    max_iters=max_iters,
                )
            if loss < best_loss:
                best_lambda_index = i
                best_gamma_index = j
                best_loss = loss

    best_lambda = lambdas[best_lambda_index]
    best_gamma = gammas[best_gamma_index]
    return best_lambda, best_gamma, best_loss


def k_fold_cross_validation_reg_log_backtrack(
    y, tx, k_fold, lambdas, seed=12, initial_w=None, max_iters=1000, verbose=False
):
    """Cross validation over regularisation parameter lambda of regularized logistic regression.

    Args:
        y:         shape=(N,)
        tx:        shape=(N,D)
        k_fold:    integer, the number of folds
        lambdas:   shape = (p, ) where p is the number of values of lambda to test
        seed:      randomness replication
        initial_w: w at iteration 0 in gradient descent
        max_iters: max iterations of gradient descent
        verbose: if true, print information
    Returns:
        best_lambda : scalar, value of the best lambda
        best_gamma : scalar, value of the best gamma
        best_loss : scalar, the associated root mean squared error for the best lambda
    """

    k_indices = __build_k_indices(y, k_fold, seed)
    best_lambda_index = 0
    best_loss = np.inf

    for i, lambda_ in enumerate(lambdas):
        print("==================")
        print(f"Starting validation for lambda={lambda_} ({i + 1}/{len(lambdas)}).")
        loss = 0.0
        for k in range(k_fold):
            loss += __cross_validation_one_iter_reg_log_backtrack(
                y,
                tx,
                k_indices,
                k,
                lambda_,
                initial_w=initial_w,
                max_iters=max_iters,
                verbose=verbose,
            )
        if loss < best_loss:
            best_lambda_index = i
            best_loss = loss

    best_lambda = lambdas[best_lambda_index]
    return best_lambda, best_loss


def k_fold_cross_validation_stoch_reg_log_backtrack(
    y, tx, k_fold, lambdas, seed=12, initial_w=None, max_iters=3000
):
    """Cross validation over regularisation parameter lambda of stochastic regularized logistic regression.

    Args:
        y:         shape=(N,)
        tx:        shape=(N,D)
        k_fold:    integer, the number of folds
        lambdas:   shape = (p, ) where p is the number of values of lambda to test
        seed:      randomness replication
        initial_w: w at iteration 0 in gradient descent
        max_iters: max iterations of gradient descent
    Returns:
        best_lambda : scalar, value of the best lambda
        best_gamma : scalar, value of the best gamma
        best_loss : scalar, the associated root mean squared error for the best lambda
    """

    k_indices = __build_k_indices(y, k_fold, seed)
    best_lambda_index = 0
    best_loss = np.inf

    for i, lambda_ in enumerate(lambdas):
        print("==================")
        print(f"Starting validation for lambda={lambda_} ({i}).")
        loss = 0.0
        for k in range(k_fold):
            print(f"Commencing fold {k + 1}/{k_fold}.")
            loss += __cross_validation_one_iter_stoch_reg_log_backtrack(
                y, tx, k_indices, k, lambda_, initial_w=initial_w, max_iters=max_iters
            )
        if loss < best_loss:
            best_lambda_index = i
            best_loss = loss

    best_lambda = lambdas[best_lambda_index]
    return best_lambda, best_loss


def __build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> __build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def __cross_validation_one_iter_reg_log(
    y, x, k_indices, k, lambda_, gamma, initial_w=None, max_iters=3000
):
    """Return the loss of regularized logistic regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar
        gamma:      scalar

    Returns:
        test loss
    """

    if initial_w is None:
        initial_w = np.zeros((x.shape[1],))

    y_te = y[k_indices[k]]
    x_te = x[k_indices[k]]

    y_tr = np.delete(y, k_indices[k])
    x_tr = np.delete(x, k_indices[k], axis=0)

    w, _ = imp.reg_logistic_regression(
        y_tr, x_tr, lambda_, initial_w=initial_w, max_iters=max_iters, gamma=gamma
    )

    loss_te = imp.compute_loss_logistic(y_te, x_te, w)
    return loss_te


def __cross_validation_one_iter_reg_log_backtrack(
    y, x, k_indices, k, lambda_, initial_w=None, max_iters=3000, verbose=False
):
    """Return the loss of regularized backtracking logistic regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar

    Returns:
        test loss
    """

    if initial_w is None:
        initial_w = np.zeros((x.shape[1],))

    y_te = y[k_indices[k]]
    x_te = x[k_indices[k]]

    y_tr = np.delete(y, k_indices[k])
    x_tr = np.delete(x, k_indices[k], axis=0)

    w, _ = reg_logistic_regression_backtracking(
        y_tr, x_tr, lambda_, initial_w=initial_w, max_iters=max_iters, verbose=verbose
    )

    loss_te = imp.compute_loss_logistic(y_te, x_te, w)
    return loss_te


def __cross_validation_one_iter_stoch_reg_log_backtrack(
    y, x, k_indices, k, lambda_, initial_w=None, max_iters=3000
):
    """Return the loss of stochastic regularized backtracking logistic regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar

    Returns:
        test loss
    """

    if initial_w is None:
        initial_w = np.zeros((x.shape[1],))

    y_te = y[k_indices[k]]
    x_te = x[k_indices[k]]

    y_tr = np.delete(y, k_indices[k])
    x_tr = np.delete(x, k_indices[k], axis=0)

    w, _ = stochastic_reg_backtracking_logistic_regression(
        y_tr, x_tr, lambda_, initial_w=initial_w, max_iters=max_iters
    )

    loss_te = imp.compute_loss_logistic(y_te, x_te, w)
    return loss_te
