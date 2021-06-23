import numpy as np
from numpy.random import default_rng
from ProblemModule.utils import jaccard_index, overlap_coefficient, solution_space_similarity

rng = default_rng(42)

dx_A = [-0.0005,   0.003]
dy_A = [-0.0025,   0.0025]
Dx_A = [8.0,   1e308]
Dy_A = [8.0,   1e308]
v_A = [16.0,   1e308]
res_A = [0.0, 0.003]

dx_B = [-0.001,   0.001]
dy_B = [-0.005,   0.005]
Dx_B = [8.0,   1e308]
Dy_B = [8.0,   1e308]
v_B = [16.0,   1e308]
res_B = [0.0, 0.001]

A = np.array([
    dx_A,
    dy_A,
    v_A,
    Dx_A,
    Dy_A,
    res_A,
]).T

B = np.array([
    dx_B,
    dy_B,
    v_B,
    Dx_B,
    Dy_B,
    res_B,
]).T

# C = np.vstack((A, B))
# C = np.array([np.minimum(A[0, :], B[0, :]),
#               np.maximum(A[1, :], B[1, :])])
mins = np.minimum(A[0, :], B[0, :])
maxs = np.maximum(A[1, :], B[1, :])

# A_normal = ((A - C.min(axis=0)) / (C.max(axis=0) - C.min(axis=0)))
# B_normal = ((B - C.min(axis=0)) / (C.max(axis=0) - C.min(axis=0)))
# C_normal = ((C - C.min(axis=0)) / (C.max(axis=0) - C.min(axis=0)))

A_normal = ((A - mins) / (maxs - mins))
B_normal = ((B - mins) / (maxs - mins))

# Omega = np.array((C_normal.min(axis=1), C_normal.max(axis=1))).T

n_dim = A.shape[1]
n_pts = 100

samples = rng.random([n_dim, n_pts]).T

mask_A = np.all(np.logical_and(samples >= A_normal.min(axis=0), samples <= A_normal.max(axis=0)), axis=1)
mask_B = np.all(np.logical_and(samples >= B_normal.min(axis=0), samples <= B_normal.max(axis=0)), axis=1)

print(jaccard_index(mask_A, mask_B))
print(overlap_coefficient(mask_A, mask_B))
print(solution_space_similarity(mask_A, mask_B))

pass
