import numpy as np


def preprocess(x, y, ids):
    """
    Apply all the preprocessing pipeline to data x and labels y. In particular, preprocess
    will divide x, y and ids according to the number of jets in x

    Args:
        x: data
        y: labels
        ids: ids of data samples
    Returns:
        preprocessed (x, y, ids) as 3 arrays of length 4, one for each possible number of jets
    """

    column_names_str = (
        "DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,"
        "DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,"
        "DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,"
        "PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,"
        "PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,"
        "PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt"
    )
    column_names = column_names_str.split(",")
    col_to_idx = {column_names[i]: i for i in range(len(column_names))}

    angle_columns = [col for col in column_names if "phi" in col]
    x = __replace_angles_with_cos_sin(x, angle_columns, col_to_idx)

    mask_0 = x.T[col_to_idx["PRI_jet_num"]] == 0
    x_0 = x[mask_0]
    y_0 = y[mask_0]
    id_0 = ids[mask_0]

    mask_1 = x.T[col_to_idx["PRI_jet_num"]] == 1
    x_1 = x[mask_1]
    y_1 = y[mask_1]
    id_1 = ids[mask_1]

    mask_2 = x.T[col_to_idx["PRI_jet_num"]] == 2
    x_2 = x[mask_2]
    y_2 = y[mask_2]
    id_2 = ids[mask_2]

    mask_3 = x.T[col_to_idx["PRI_jet_num"]] == 3
    x_3 = x[mask_3]
    y_3 = y[mask_3]
    id_3 = ids[mask_3]

    xs = [x_0, x_1, x_2, x_3]
    ys = [y_0, y_1, y_2, y_3]
    ids = [id_0, id_1, id_2, id_3]

    txs = []

    for i, xi in enumerate(xs):
        discarded_columns = [
            column_names.index("PRI_jet_num")
        ]  # discarded as it contains integers
        for angle_col in angle_columns:
            discarded_columns.append([column_names.index(angle_col)])
            # not needed anymore as we have replaced them with cos/sin

        threshold = 0.5  # if a column contains more than threshold * 100 % of missing_values, discard it

        discarded_columns.extend(__cols_too_many_missing_values(xi, threshold))

        mask = [i not in discarded_columns for i in range(xi.shape[1])]
        xi = xi.T[mask].T

        xi = __replace_missing_with_mean(xi)
        xi = __standardize(xi)
        txi = __concat_bias(xi)
        txs.append(txi)

        ys[i] = __put_in_0_1(ys[i])

    return txs, ys, ids


def __put_in_0_1(y):
    """
    Args:
        y: vector in {-1, 1}^N
    Returns:
        y translated to {0, 1}^N
    """
    return (y + 1) / 2


def __concat_bias(x):
    """
    Concatenates a column of 1s to data to account for bias term

    Args:
        x: data
    Returns:
        new array with additional column
    """
    return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)


def __standardize(tx):
    """
    Args:
        tx: array to standardize
    Returns:
        Standardized version of x, with 0 mean and 1 std
    """
    centered_data = tx - np.mean(tx, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    return std_data


def __cols_too_many_missing_values(
    x, threshold, missing_value_identifier=-999, zero_std_threshold=1e-5
):
    """
    Determine columns with too many missing values

    Args:
        x: data
        column_names: all names of columns
        threshold: ratio of missing values above which there are too many missing values
        missing_value_identifier
    Returns:
        column names with too many missing values
    """
    discarded_columns = []

    nb_samples = x.shape[0]
    for idx, col in enumerate(x.T):
        count_missing = np.count_nonzero(col == missing_value_identifier)
        if (
            float(count_missing) / nb_samples > threshold
            or np.std(col) < zero_std_threshold
        ):
            discarded_columns.append(idx)

    return discarded_columns


def __replace_angles_with_cos_sin(x, angle_columns, col_to_idx):
    """
    Replaces angle columns in [-pi, pi] to 2 columns: their cosine and sine

    Args:
        x: data
        angle_columns: column names of angle columns
        col_to_idx: map column names to indices
    Returns:
        data with new columns, appended to the right
    """
    for angle_col in angle_columns:
        idx = col_to_idx[angle_col]
        x = np.concatenate(
            [x, np.cos(x.T[idx]).reshape((-1, 1)), np.sin(x.T[idx]).reshape((-1, 1))],
            axis=1,
        )
    return x


def __replace_missing_with_mean(x, missing_value_identifier=-999.0):
    """
    Replaces all missing values in data by mean value of same column

    Args:
        x: initial data
        missing_value_identifier
    Returns:
        x with replaced missing values
    """
    for idx, col in enumerate(x.T):
        mask = col != missing_value_identifier
        col_no_missing_vals = col[col != missing_value_identifier]
        col_mean = np.mean(col_no_missing_vals)
        x.T[idx][~mask] = col_mean
    return x
