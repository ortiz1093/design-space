import pandas as pd
import numpy as np
from numpy.random import default_rng
import seaborn as sns
import matplotlib.pyplot as plt


# def split_df_numerical_categorical(df):
#     df_numeric = df.select_dtypes(include=[int, float])
#     df_categorical = df.select_dtypes(exclude=[int, float]).astype('category')

#     return df_numeric, df_categorical


# def categorical_diff(df_categorical):
#     df_categorical_diffs = pd.DataFrame()

#     for column in df_categorical.columns:
#         df_categorical_diffs[column] = df_categorical[column].cat.codes.diff()
    
#     return df_categorical_diffs.map(lambda x: np.sign(x)**2)


# def scaled_diff(df_numeric):
#     df_scaled = (df_numeric - df_numeric.min(0)) / (df_numeric.max(0) - df_numeric.min(0))
#     return df_scaled.diff()


# def mixed_dtype_diff(df):
#     df_numeric, df_categorical = split_df_numerical_categorical(df)

#     df_numeric_diffs = scaled_numeric_diff(df_numeric)
#     df_categorical_diffs = categorical_diff(df_categorical)

#     return pd.concat([df_numeric_diffs, df_categorical_diffs], axis=1)[1:]

# def mixed_dataframe_norms(df):
#     df_diffs = mixed_dtype_diff(df)
#     sum_squared_diffs = (df_diffs**2).sum(1)

#     return np.sqrt(sum_squared_diffs)


def mixed_data_norms(X_uns, categorical_cols):
    cols = np.arange(X_uns.shape[1])
    cat_idx = [True if i in categorical_cols else False for i in cols]
    X = (X_uns - X_uns.min(0)) / (X_uns.max(0) - X_uns.min(0))
    X = X_uns

    norms = []
    for row in X:
        abs_diffs = np.abs(X - row)
        abs_diffs[:, cat_idx] = np.sign(abs_diffs[:, cat_idx])
        norms.append(np.sqrt(np.sum(abs_diffs**2, axis=1)))

    return np.array(norms)


def mixed_data_dmap(X, categorical_cols, epsilon):
    norms = mixed_data_norms(X, categorical_cols)
    diffusions = np.exp(-norms / epsilon)
    scales = np.sum(diffusions, axis=0)**.5
    normalized = diffusions / (scales[:, None] * scales[None, :])
    values, vectors = np.linalg.eigh(normalized)
    basis_vectors = vectors / scales[:, None]
    basis = basis_vectors * values[None, :]

    return np.flip(basis,axis=1), np.flip(values)


rng = default_rng(42)

s = [21.0, 27.0]
q = [2.0, 10.0]
t = [0.1, 0.2]
Y = [9e6, 12e6]
G = [10.0, 18.00]
mu = [0.0625, 0.125, 0.25]
p = [1.0, 3.0]
n = [15, 50, 1]
Motor = [chr(i) for i in range(65,75)]

N = 200

df = pd.DataFrame(dict(
    s=rng.random(N) * np.diff(s) + min(s),
    q=rng.random(N) * np.diff(q) + min(q),
    t=rng.random(N) * np.diff(t) + min(t),
    Y=rng.random(N) * np.diff(Y) + min(Y),
    G=rng.random(N) * np.diff(G) + min(G),
    mu=rng.choice(mu, N),
    p=rng.random(N) * np.diff(p) + min(p),
    n=rng.integers(min(n), max(n), N),
    Motor=rng.choice(Motor, N)
))

df.Motor = df.Motor.astype('category').cat.codes

# TODO: Determine how to treat discrete axes with non-linear intervals, e.g. mu (med-priority)
# TODO: Determine how to treat discrete axes with uneven intervals (low-priority)
# TODO: Implement distance function for mixed-data (med-priority)

sns.pairplot(df)

cat_cols = [8]
Y, sigmas = mixed_data_dmap(df.to_numpy(), cat_cols, 30)


new_df = pd.DataFrame(Y[:, :5])


sns.pairplot(new_df)
plt.show()
pass