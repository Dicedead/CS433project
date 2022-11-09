import numpy as np

from helpers import *
from preprocessing import *
from implementations import *
from datetime import datetime

"""
Generate a submission in data/submissions.

Assumes: 4 vectors of weights w are stored in "data/weights.txt", and the test data is in "data/test.csv".
"""

now = datetime.now()
datetime_str = now.strftime("%Y_%m_%d-%H_%M_%S")

data_test = "~/../data/test.csv"
weights_path = "~/../data/weights.txt"

y_te, x_te, id_te = load_csv_data(data_test, sub_sample=False)

txs_te, ys_te, ids_te = preprocess(x_te, y_te, id_te)
ws = read_weights(weights_path, len(txs_te))
y_preds = []

for i, w in enumerate(ws):
    y_pred = logistic_regression_classify(txs_te[i], w)
    y_preds.append(y_pred)

file_name = "data/submissions/submission_" + datetime_str + ".csv"
create_csv_submission(np.concatenate(ids_te), np.concatenate(y_preds), file_name)
print("Done!")
