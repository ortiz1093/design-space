from ProblemModule import Design
from time import time


def normal():
    num_pts = 1_000
    t0 = time()

    reqs_file = '/root/ThesisCode/3DP_design_files/3DP_reqs.json'
    vars_file = '/root/ThesisCode/3DP_design_files/3DP_design_vars__new_motors.json'
    func_file = "/root/ThesisCode/3DP_design_files/3DP_funcs.json"

    print("#"*45)
    print("Initializing Design Object...")
    test_design = Design()

    print("Constructing Design Object from files...")
    print("\tRequirements...")
    test_design.load_requirements_from_json(reqs_file)
    print("\tDesign Variables...")
    test_design.append_variables_from_json(vars_file)
    print("\tMap...")
    test_design.set_map_from_json(func_file)
    print("Building Constraint Space...")
    test_design.build_constraint_space()
    test_design.plot_problem()
    print("Building Form Space...")
    test_design.build_form_space(N=num_pts)
    # test_design.plot_solutions(show_fails=True)
    test_design.plot_solutions(show_fails=False)

    print("#"*45)

    t = time() - t0
    print(f"\nDisplaying {num_pts} points")
    print(f"Elapsed time: {round(t, 2)}s")


def reduction():
    num_pts = 1_000

    # Specify filenames
    reqs_file = '/root/ThesisCode/3DP_design_files/3DP_reqs.json'
    vars_file = '/root/ThesisCode/3DP_design_files/3DP_design_vars__new_motors.json'
    func_file = "/root/ThesisCode/3DP_design_files/3DP_funcs.json"

    # Create design from files
    test_design = Design()
    test_design.load_requirements_from_json(reqs_file)
    test_design.append_variables_from_json(vars_file)
    test_design.set_map_from_json(func_file)

    # Generate form space
    test_design.build_form_space(N=num_pts)
    test_design.plot_solutions(full_space=False, show_fails=False)


def values():
    num_pts = 1_000

    # Specify filenames
    reqs_file = '/root/ThesisCode/3DP_design_files/3DP_reqs.json'
    vars_file = '/root/ThesisCode/3DP_design_files/3DP_design_vars__new_motors.json'
    func_file = "/root/ThesisCode/3DP_design_files/3DP_funcs.json"

    # Create design from files
    test_design = Design()
    test_design.load_requirements_from_json(reqs_file)
    test_design.append_variables_from_json(vars_file)
    test_design.set_map_from_json(func_file)

    # Generate form space
    test_design.build_form_space(N=num_pts)
    test_design.set_value_function('dx + q')
    test_design.compute_point_values()
    test_design.apply_value_gradient()
    # test_design.plot_solutions(full_space=True, show_fails=False)
    test_design.plot_solutions(full_space=True, show_fails=False, show_gradient=True)
    pass


import seaborn as sns
import pandas as pd
import numpy as np
import plom.plom_v4_4 as plom
from plom.plom_utils import setupArgs
import plotly.graph_objects as go


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


def reduction_dmaps():
    num_pts = 1_000

    # Specify filenames
    reqs_file = '/root/ThesisCode/3DP_design_files/3DP_reqs.json'
    vars_file = '/root/ThesisCode/3DP_design_files/3DP_design_vars__new_motors.json'
    func_file = "/root/ThesisCode/3DP_design_files/3DP_funcs.json"

    # Create design from files
    test_design = Design()
    test_design.load_requirements_from_json(reqs_file)
    test_design.append_variables_from_json(vars_file)
    test_design.set_map_from_json(func_file)

    # Generate form space
    test_design.build_form_space(N=num_pts)
    
    # Diffusion maps
    cat_cols = [8]
    df = test_design.form_space.points_df
    Y, _ = mixed_data_dmap(df.to_numpy(), cat_cols, 30)

    # Plotting seaborn
    new_df = pd.DataFrame(Y[:, :5])
    g = sns.pairplot(new_df, diag_kind='kde', corner=True, aspect=1, height=1)
    g.map_lower(sns.kdeplot, levels=4, color=".2")
    # plt.show()

    # Plotting plotly
    fig = go.Figure(data=[
        go.Scatter3d(x=Y[:, 1], y=Y[:, 2], z=Y[:, 3], mode='markers')
    ])
    fig.show()

    pass


def gauge_test():
    num_pts = 10_000

    # Specify filenames
    reqs_file = '/root/ThesisCode/3DP_design_files/3DP_reqs.json'
    vars_file = '/root/ThesisCode/3DP_design_files/3DP_design_vars__new_motors.json'
    func_file = "/root/ThesisCode/3DP_design_files/3DP_funcs.json"
    criteria_file = "/root/ThesisCode/3DP_design_files/3DP_criteria.json"

    # Create design from files
    test_design = Design()
    test_design.load_requirements_from_json(reqs_file)
    test_design.append_variables_from_json(vars_file)
    test_design.set_map_from_json(func_file)
    test_design.set_criteria_map_from_json(criteria_file)

    # Generate form space
    test_design.build_form_space(N=num_pts)
    test_design.build_gauge_space()
    test_design.plot_criteria()

    # build guage space

    pass


def similarity_test():

    # Specify filenames
    reqs_fileA = '/root/ThesisCode/3DP_design_files/3DP_reqs.json'
    reqs_fileB = '/root/ThesisCode/3DP_design_files/3DP_reqsB.json'
    vars_file = '/root/ThesisCode/3DP_design_files/3DP_design_vars__new_motors.json'
    func_file = "/root/ThesisCode/3DP_design_files/3DP_funcs.json"

    # Create design from files
    designA = Design()
    designA.load_requirements_from_json(reqs_fileA)
    designA.append_variables_from_json(vars_file)
    designA.set_map_from_json(func_file)
    
    designB = Design()
    designB.load_requirements_from_json(reqs_fileB)
    designB.append_variables_from_json(vars_file)
    designB.set_map_from_json(func_file)

    # Similarity
    n_pts = 100_000
    soln_similarity = designA.solution_space_similarity(designB, num_samples=n_pts)
    prblm_similarity = designA.problem_space_similarity(designB, num_samples=n_pts)
    dS = 1 - soln_similarity
    dP = 1 - prblm_similarity
    dSdP = designA.sensitivity(designB, num_samples=n_pts)

    print('Soln Similarity: ', soln_similarity)
    print('Prblm Similarity: ', prblm_similarity)
    print('dS: ', dS)
    print('dP: ', dP)
    print('Sensitivity: ', dSdP)

    designA.build_constraint_space()
    designA.plot_problem()
    designA.build_form_space(N=n_pts)
    designA.plot_solutions(show_fails=False)

    pass


def models():
    num_pts = 1_000

    reqs_file = '/root/ThesisCode/3DP_design_files/3DP_reqs_w_models.json'
    vars_file = '/root/ThesisCode/3DP_design_files/3DP_design_vars_w_gantry_model.json'
    func_file = "/root/ThesisCode/3DP_design_files/3DP_funcs_models.json"

    print("#"*45)
    print("Initializing Design Object...")
    test_design = Design()
    print("Constructing Design Object from files...")
    print("\tRequirements...")
    test_design.load_requirements_from_json(reqs_file)
    print("\tDesign Variables...")
    test_design.append_variables_from_json(vars_file)
    print("\tMap...")
    test_design.set_map_from_json(func_file)
    print("Building Constraint Space...")
    test_design.build_constraint_space()
    # test_design.plot_problem()
    print("Building Form Space...")
    test_design.build_form_space(N=num_pts)
    # test_design.plot_solutions(show_fails=True)

    pt = dict(s=20.0, q=8.1, t=0.5, Y=8.9e6, G=18, w=0.26, h=0.8, t_f=0.08, t_w=0.2, mu=0.5, p=2.0, n=50,
                L=3e-3, V=3.4, Amp=1.0, phi=1.8)
    print(test_design.point_conformity(pt))

# values()
# reduction()
# normal()
# reduction_dmaps()
similarity_test()
# gauge_test()
# models()
# plt.show()
pass
