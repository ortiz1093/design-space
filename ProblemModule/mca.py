import pandas as pd
import numpy as np
from scipy.linalg import fractional_matrix_power, diagsvd


def categorize1D(ndarr):
    arr_unq = np.sort(np.unique(ndarr))
    diffs = np.diff(arr_unq)
    three_sigma = 3 * np.std(diffs)
    splits = np.argwhere(diffs > three_sigma)

    labels = ndarr.copy()
    label = 1
    prev = min(arr_unq)-1

    for split in splits:
        labels[np.logical_and(prev < ndarr, ndarr <= arr_unq[split])] = label
        label += 1
        prev = arr_unq[split]

    labels[ndarr > arr_unq[splits[-1]]] = label

    return labels


def multiple_correspondence_analysis(dataframe):
    """
    See https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf for full explanation
    """

    # Convert the dataframe to its Indicator Matrix (I)
    dataframe = dataframe.astype('int')
    dataframe = dataframe - dataframe.min(0)
    X = pd.get_dummies(data=dataframe, columns=dataframe.columns).to_numpy()

    # Compute the probability matrix (Z)
    N = X.sum()
    Z = X/N

# Compute the column and row sums of the indicator matrix
    c = Z.sum(axis=0)
    r = Z.sum(axis=1)

    # Convert the sums into diagonal matrices
    D_c = np.diag(c)
    D_r = np.diag(r)

    # Compute the MCA matrix
    # M = fractional_matrix_power(D_r, -0.5) \
    #     @ (Z - np.outer(r, c)) \
    #     @ fractional_matrix_power(D_c, -0.5)

    M = np.sqrt(np.diag(r**-1)) \
        @ (Z - np.outer(r, c)) \
        @ np.sqrt(np.diag(c**-1))

    # Compute the SVD of the MCA matrix
    _, S, Qt = np.linalg.svd(M, full_matrices=False)

    # Sort S and Qt
    Q = np.fliplr(Qt.T[:, S.argsort()])
    S = np.flip(np.sort(S))

    # Compute the column factor scores of the MCA matrix
    # G = fractional_matrix_power(D_c, -0.5) @ Q @ diagsvd(S, *X.shape)
    G = np.sqrt(np.diag(c**-1)) @ Q @ diagsvd(S, *X.shape)

    return X, G, S


def factor_space_projection(dataframe, n_dim='auto', scale=1):
    X, G, S = multiple_correspondence_analysis(dataframe)

    _, K = dataframe.shape
    _, J = X.shape

    # Get eigenvalues of indicator matrix
    Lambda = S**2
    # Lambda.sort()
    # Lambda = np.flip(Lambda)

    # Compute the Greenacre correction factors
    cScale = K / (K - 1)
    cLamb = (cScale * (Lambda - (1 / K)))**2
    cLambda = cLamb[Lambda > 1 / K]

    # Compute the corrected contributions (explained inertia) of each axis
    I_bar = cScale * (sum(Lambda**2) - (J - K)/K**2)
    cTau = cLambda / I_bar

    # Determine the number of dimension (if not provided
    if n_dim == 'auto':
        n_dim = len(cTau)

    # Return
    return X @ G[:, :n_dim] / S[:n_dim] / scale, cTau.round(3)


if __name__ == '__main__':
    df1 = pd.DataFrame({'Fruity1': [0, 1, 1, 1, 0, 0],
                        'Woody1': [2, 1, 0, 0, 2, 1],
                        'Coffee1': [1, 0, 0, 0, 1, 1],
                        'RedFruit2': [0, 1, 1, 1, 0, 0],
                        'Roasted2': [1, 0, 0, 0, 1, 1],
                        'Vanillin2': [2, 1, 0, 0, 2, 1],
                        'Woody2': [1, 0, 0, 0, 1, 1],
                        'Fruity3': [1, 1, 1, 0, 0, 0],
                        'Butter3': [1, 0, 0, 0, 1, 1],
                        'Woody3': [1, 0, 0, 0, 1, 1]})

    # df2 = pd.DataFrame({'Q1': [2, 1, 2, 3, 2, 2, 1, 2, 2],
    #                     'Q2': [1, 1, 2, 2, 2, 3, 2, 3, 2],
    #                     'Q3': [1, 2, 3, 1, 1, 2, 2, 3, 1],
    #                     'Q4': [3, 2, 1, 1, 1, 1, 3, 3, 2]})

    X2, G2, _ = multiple_correspondence_analysis(df1)
    X_proj, contrib = factor_space_projection(df1)
    pass
