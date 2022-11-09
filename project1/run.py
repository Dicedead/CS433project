import os

from additional_implementations import reg_logistic_regression_backtracking
from helpers import load_csv_data, create_csv_submission
from implementations import logistic_regression_classify
from preprocessing import preprocess

from metrics import *

"""
Outputs final submission CSV file. Assumes data folder contains train.csv and test.csv.
"""
data_train = "data/train.csv"
data_test = "data/test.csv"
csv_sub_filename = "data/submissions/submission_final.csv"


def run():
    if not os.path.exists(data_train):
        print(
            "Cannot find the file data/train.csv, please provide the script with the training dataset there."
        )
        return

    if not os.path.exists(data_test):
        print(
            "Cannot find the file data/test.csv, please provide the script with the testing dataset there."
        )
        return

    print("Starting loading...")
    y_tr, x_tr, id_tr = load_csv_data(data_train, sub_sample=False)
    y_te, x_te, id_te = load_csv_data(data_test, sub_sample=False)
    print("Loading finished. Starting preprocessing...")
    txs_tr, ys_tr, ids_tr = preprocess(x_tr, y_tr, id_tr)
    txs_te, ys_te, ids_te = preprocess(x_te, y_te, id_te)
    print("Preprocessing done.")

    lambda_ = 0.001
    max_iters = 3000

    y_preds = []

    for i in range(len(txs_tr)):
        print("###########################")
        print(f"Case: jet_num = {i}")
        initial_w = np.zeros((txs_tr[i].shape[1],))
        w, loss = reg_logistic_regression_backtracking(
            ys_tr[i], txs_tr[i], lambda_, initial_w, max_iters, verbose=True
        )
        y_pred = logistic_regression_classify(txs_te[i], w)
        y_preds.append(y_pred)
        y_pred_tr = logistic_regression_classify(txs_tr[i], w)
        print(f"Training accuracy: {accuracy(y_pred_tr, ys_tr[i]) * 100}%")
        print(f"Training F1: {f1_score(y_pred_tr, ys_tr[i])}")

    y_pred = np.concatenate(y_preds)
    id_te = np.concatenate(ids_te)
    create_csv_submission(id_te, y_pred, csv_sub_filename)
    print("Done!")


run()
