import pandas as pd
import numpy as np
from scipy.linalg import fractional_matrix_power, diagsvd


def continuous2category():
    # TODO: Use clustering algo to map each continuous variable into categories
    pass


def multiple_correspondence_analysis(dataframe):
    X = pd.get_dummies(data=dataframe, columns=dataframe.columns).to_numpy()
    c = X.sum(axis=0)
    r = X.sum(axis=1)
    D_c = np.diag(c)
    D_r = np.diag(r)
    N = X.sum()
    Z = X/N

    M = fractional_matrix_power(D_r, -0.5) \
        @ (Z - np.outer(r, c)) \
        @ fractional_matrix_power(D_c, -0.5)

    U, S, Vt = np.linalg.svd(M)

    G = fractional_matrix_power(D_c, -0.5) @ Vt.T @ diagsvd(S, *X.shape).T

    return X, G, S


def factor_space_projection(dataframe, N_dim='all', scale=1):
    X, G, S = multiple_correspondence_analysis(dataframe)

    if N_dim == 'all':
        N_dim = G.shape[1]

    return X @ G[:, :N_dim] / S[:N_dim] / scale


if __name__ == '__main__':
    df2 = pd.DataFrame({'Q1': [2, 1, 2, 3, 2, 2, 1, 2, 2],
                        'Q2': [1, 1, 2, 2, 2, 3, 2, 3, 2],
                        'Q3': [1, 2, 3, 1, 1, 2, 2, 3, 1],
                        'Q4': [3, 2, 1, 1, 1, 1, 3, 3, 2]})

    X2, G2, _ = multiple_correspondence_analysis(df2)
    X_proj = factor_space_projection(df2)
    pass
