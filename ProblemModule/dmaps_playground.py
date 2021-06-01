import plotly.graph_objects as go
import numpy as np
from numpy.random import default_rng
from numpy.linalg import norm, inv
from scipy.linalg import eig, eigh
from tqdm import tqdm
from joblib import Parallel, delayed
import sys
import seaborn as sns
import pandas as pd
import plom.plom_v4_4 as plom
from plom.plom_utils import setupArgs


rng = default_rng(42)


def rotate2D(X, deg, axis='origin'):
    vertical = True if X.shape[1]==2 else False
    X = X.T if vertical else X

    rad = np.radians(deg)
    R = np.array([[np.cos(rad), -np.sin(rad)],
                  [np.sin(rad), np.cos(rad)]])
    
    X_rot = R @ X

    return X_rot.T if vertical else X_rot


# def minmax_cols(X):
#     n_rows, n_cols = X.shape
#     col_mins = np.tile(X.min(0), [n_rows, 1])
#     col_maxs = np.tile(X.max(0), [n_rows, 1])

#     return (X - col_mins) / (col_maxs - col_mins)


# def diffusion_kernel(x0, x1, eps):
#     return np.exp(-norm(x0 - x1) / eps)


# def compute_distance_matrix_element(i, j, X, eps):
#     return i, j, diffusion_kernel(X[i, :], X[j, :], eps)


# def distance_matrix(X, n_dim, eps):
#     assert n_dim in X.shape, "X does not match the specified number of dimensions"
    
#     X = X if X.shape[1] == n_dim else X.T
#     n_pts = X.shape[0]

#     K = np.empty([n_pts, n_pts])
#     # for i in tqdm(range(n_pts), desc='Building matrix K', position=0):
#     #     for j in range(i, n_pts):
#     #         K[i, j] = K[j, i] = diffusion_kernel(X[i, :], X[j, :], eps)
#     inputs = tqdm([(i, j) for i in range(num_pts) for j in range(i, num_pts)])
#     params = [X, eps]
#     elements = Parallel(n_jobs=8)(delayed(compute_distance_matrix_element)(*i, *params) for i in inputs)
#     for i, j, K_ij in elements:
#         K[i, j] = K[j, i] = K_ij
    
#     return K


# def dmaps(X, n_dim, eps=10):
#     """
#     Computes the diffusion map of X and returns the transformed dataset. Each column of Y is one point in the dataset.
#     Rows in Y are sorted by descending importance, i.e. the first coordinate/row of each point is the most import, etc.
#     """
#     assert X.shape[1] == n_dim, "X must be vertical array (samples->rows, features->columns)"
#     X_scaled = minmax_cols(X)

#     t1 = time()
#     K = distance_matrix(X_scaled, n_dim, eps)
#     # print(f'Distance Matrix time: {round(time() - t1, 2)}s')
#     D = np.diag(K.sum(1))

#     t2 = time()
#     P = inv(D) @ K
#     # print(f'Inversion time: {round(time() - t2, 2)}s')
    
#     t3 = time()
#     # w, v_left = eigh(P)
#     w, v_left = eig(P, left=True, right=False)
#     # print(f'Eigenvalues time: {round(time() - t3, 2)}s')
    
#     w = np.real(w)
#     i_sort = np.flip(w.argsort())
#     w = w[i_sort]
#     v_left = v_left[i_sort]
    
#     return np.diag(w) @ v_left.T

from time import time

t0 = time()

num_grps = 5

num_pts = 500
num_turns = 4

r = 1
h = 3
theta = np.linspace(0, 2 * np.pi * num_turns, num_pts)

x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.linspace(-h, h, num_pts)

X = np.array([x, y, z]).T

grp_size = num_pts // num_grps
grp_labels = np.array([[i]*grp_size for i in range(num_grps)]).flatten()[:num_pts]

t_plom = time()
epsilon = 30
args_plom = setupArgs(X, epsilon, sampling=True)
plom_dict = plom.initialize(**args_plom)
plom.run(plom_dict)
print(f'PLoM took {round(time() - t_plom, 2)}s')

X_plom = plom_dict['dmaps']['basis']
plomDF = pd.DataFrame(X_plom[:,1:6])
plomDF['group'] = grp_labels
# sns.pairplot(plomDF.loc[:,[0,2,'group']], hue='group', palette='hls')
sns.pairplot(plomDF, hue='group', palette='hls')

# import matplotlib.pyplot as plt
# plt.show()

# quit()
Z = np.array([
    [0.0001, 0.0001, 0.0002, -1, 1],
    [-0.0002, -0.0001, 0.0002, 1, -1],
    [0.0002, 0.0002, 0, 0, 0]
])
X_new = plom._inverse_dmaps(Z, X_plom[:,1:6])

fig = go.Figure(data=[
    go.Scatter3d(x=X[:, 0], y=X[:, 1], z=X[:, 2]),
    go.Scatter3d(x=X_new[:, 0], y=X_new[:, 1], z=X_new[:, 2])
]
)
fig.show()