"""Some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    y_pred = 2 * y_pred - 1

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def read_weights(txt_path, nb_weights_to_read):
    """
    Read array of vectors of weights w from a text file

    Args:
        txt_path: path of text file
        nb_weights_to_read: number of vectors to read
    Returns:
        array of numpy nd-arrays
    """
    file = open(txt_path, "r")
    ws = []
    for _ in range(nb_weights_to_read):
        w = file.readline()
        ws.append(np.array(w.split(" "), dtype=np.float_))
    file.close()
    return ws


def write_weights(txt_path, ws):
    """
    Write an array of vectors of weights w to a text file

    Args:
        txt_path: path of text file
        ws: array of vector of weights
    """
    file = open(txt_path, "w")
    for w in ws:
        w_str = " ".join(map(str, w)) + "\n"
        file.write(w_str)
    file.close()
