import metrics
from helpers import *
from preprocessing import *
from validation import *
from additional_implementations import *
from datetime import datetime

"""
Trains the model on training data located in data/train.csv and stores a vector of weights w in data/weights.txt.
Performs cross validation for regularised logistic regression.
"""

now = datetime.now()
datetime_str = now.strftime("%Y_%m_%d-%H_%M_%S")

data_train = "../data/train.csv"
output_path = "../data/weights.txt"

validation_path = "../validation_data/validating_" + datetime_str + ".txt"

y_tr, x_tr, id_tr = load_csv_data(data_train, sub_sample=False)
txs_tr, ys_tr, ids_tr = preprocess(x_tr, y_tr, id_tr)


def validate_reg_log_backtrack(y_tr, tx_tr, k_fold=10, lambdas=None, verbose=False):
    """
    k-fold cross validation method for regularized logistic regression with backtracking linesearch

    Args:
        y_tr: training labels
        tx_tr: training data
        k_fold: number of folds
        lambdas: values to test
        verbose
    Returns:
        best lambda
    """
    if lambdas is None:
        lambdas = [0.001, 0.00125, 0.0015, 0.00175]

    lambda_star, loss_star = k_fold_cross_validation_reg_log_backtrack(
        y_tr, tx_tr, k_fold, lambdas, verbose=verbose
    )
    print(f"Best lambda: {lambda_star}")

    validation_save = open(validation_path, "w")
    validation_save.write(
        "Regularized logistic regression with backtracking"
        + "\nLambdas: "
        + "".join(map(lambda x: str(x) + ", ", lambdas))
        + "\nLambda star: "
        + str(lambda_star)
    )
    validation_save.close()

    return lambda_star


max_iters = 1000
ws = []

for i in range(len(txs_tr)):
    print("###################################################################")
    print(f"Treating case jet num = {i}")
    lambda_ = 0.001  # validate_reg_log_backtrack(ys_tr[i], txs_tr[i], verbose=True)
    initial_w = np.zeros((txs_tr[i].shape[1],))
    w, loss = reg_logistic_regression_backtracking(
        ys_tr[i], txs_tr[i], lambda_, initial_w, max_iters, verbose=True
    )
    print(
        f"Training accuracy: {metrics.accuracy(logistic_regression_classify(txs_tr[i], w), ys_tr[i]) * 100}"
    )
    ws.append(w)

write_weights(output_path, ws)
# Best lambdas: 0.001 everywhere
