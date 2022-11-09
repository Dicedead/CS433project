import numpy as np


def count_mistakes(y_pred, y_te):
    """
    Count differences between predicted values and ground truth

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
         Count of differences
    """
    return np.count_nonzero(y_pred - y_te)


def accuracy(y_pred, y_te):
    """
    Fraction of correct predictions of total predictions

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said fraction
    """
    return 1 - (count_mistakes(y_pred, y_te) / len(y_te))


def true_positives(y_pred, y_te):
    """
    Computes the number of true positives

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said number
    """
    return np.count_nonzero(y_pred) - false_positives(y_pred, y_te)


def false_positives(y_pred, y_te):
    """
    Computes the number of false positives

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said number
    """
    return np.count_nonzero(1 - y_te[y_pred == 1])


def true_negatives(y_pred, y_te):
    """
    Computes the number of true negatives

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said number
    """
    return np.count_nonzero(y_te - 1) - false_positives(y_pred, y_te)


def false_negatives(y_pred, y_te):
    """
    Computes the number of false negatives

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said number
    """
    return np.count_nonzero(y_pred - 1) - true_negatives(y_pred, y_te)


def recall(y_pred, y_te):
    """
    Computes the true positive rate

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said rate
    """
    return float(true_positives(y_pred, y_te)) / (np.count_nonzero(y_te))


def precision(y_pred, y_te):
    """
    Computes the precision

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said number
    """
    return float(true_positives(y_pred, y_te)) / (
        true_positives(y_pred, y_te) + false_positives(y_pred, y_te)
    )


def f1_score(y_pred, y_te):
    """
    Computes the F1 score

    Args:
        y_pred: Predicted values
        y_te: Ground truth
    Returns:
        said number
    """
    return 2 / (precision(y_pred, y_te) ** (-1) + recall(y_pred, y_te) ** (-1))
