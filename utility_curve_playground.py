import json
import numpy as np
from sympy import sympify, lambdify
import plotly.graph_objects as go
from numpy.random import default_rng
import pandas as pd
import plotly.express as px


rng = default_rng(42)


def str2function(str_expression):
    sym_exp = sympify(str_expression)
    # inputs = list(sym_exp.free_symbols)
    inputs = sorted([str(symbol) for symbol in sym_exp.free_symbols])
    # print(str_expression, inputs, sep='\t')

    return lambdify(inputs, sym_exp, 'numpy')


# Function for MinMax scaling data
scale = lambda X: (X - X.min()) / (X.max() - X.min())

# Specify utility curve for each criterion
res_qlty = lambda x: 1 / x
power_qlty = lambda x: np.exp(-x)
side_qlty = lambda x: x

# Utility function
utility = lambda res_axis, pwr_axis, side_axis: \
    (3 * scale(res_qlty(res_axis)) + 1 * scale(power_qlty(pwr_axis)) + 2 * scale(side_qlty(side_axis))) / (3 + 1 + 2)

def design_A_objective_pts(N):
    # Design variable search space
    mu_opts = np.array([0.0625, 0.125, 0.25])
    p_opts = np.array([1, 3])
    n_opts = np.array([15, 50, 1])
    phi_opts = np.array([1.8])
    motor_V_opts = np.array([3.5, 4.0, 12.0, 6.0, 24.0, 6.0, 12.0, 3.4, 12.0, 3.06])
    motor_Amp_opts = np.array([1.0, 1.5, 3.6, 1.0, 0.3, 1.3, 0.7, 2.8, 0.8, 0.7])
    motor_opts = range(len(motor_V_opts))
    s_opts = np.array([21, 27])

    # Design variable random selection
    mu = rng.choice(mu_opts, N)
    p = rng.uniform(*p_opts, N)
    n = rng.choice(range(*n_opts), N)
    phi = rng.choice(phi_opts, N)
    motor = rng.choice(motor_opts, N)
    motor_V = motor_V_opts[motor]
    motor_Amp = motor_Amp_opts[motor]
    s = rng.uniform(*s_opts, N)

    # Get design criteria
    criteria_fileA = 'example_design_files/DesignA_coreXY/design_criteria_A.json'
    criteria_dictA = json.load(open(criteria_fileA, 'r'))

    # Convert string expressions into criteria map
    resolutionA = str2function(criteria_dictA['res'])
    power_usedA = str2function(criteria_dictA['pwr'])
    side_lenA = str2function(criteria_dictA['s'])

    # Objective space points    
    resA = resolutionA(mu, n, p, phi)
    pwrA = power_usedA(motor_Amp, motor_V)
    sideA = side_lenA(s)

    return resA, pwrA, sideA, utility(resA, pwrA, sideA)


def design_B_objective_pts(N):
    # Design variable search space
    mu_opts = np.array([0.0625, 0.125, 0.25])
    s_opts = np.array([21, 27])

    screw_p_opts = np.array([0.012, 0.049, 0.025, 0.125, 0.039, 0.196, 0.084, 0.197])
    screw_nu_opts = np.array([0.21, 0.89, 0.21, 0.84, 0.79, 0.85, 0.86, 0.88])
    screw_opts = range(len(screw_p_opts))
    
    phi_opts = np.array([1.8])
    motor_V_opts = np.array([3.5, 4.0, 12.0, 6.0, 24.0, 6.0, 12.0, 3.4, 12.0, 3.06])
    motor_Amp_opts = np.array([1.0, 1.5, 3.6, 1.0, 0.3, 1.3, 0.7, 2.8, 0.8, 0.7])
    motor_opts = range(len(motor_V_opts))

    # Design variable random selection
    mu = rng.choice(mu_opts, N)
    screw = rng.choice(screw_opts, N)
    p = screw_p_opts[screw]
    nu = screw_nu_opts[screw]
    phi = rng.choice(phi_opts, N)
    motor = rng.choice(motor_opts, N)
    motor_V = motor_V_opts[motor]
    motor_Amp = motor_Amp_opts[motor]
    s = rng.uniform(*s_opts, N)

    # Get design criteria
    criteria_fileB = 'example_design_files/DesignB_leadScrew/design_criteria_B.json'
    criteria_dictB = json.load(open(criteria_fileB, 'r'))

    # Convert string expressions into criteria map
    resolutionB = str2function(criteria_dictB['res'])
    power_usedB = str2function(criteria_dictB['pwr'])
    side_lenB = str2function(criteria_dictB['s'])

    # Objective space points    
    resB = resolutionB(mu, p, phi)
    pwrB = power_usedB(motor_Amp, motor_V, nu)
    sideB = side_lenB(s)
    
    return resB, pwrB, sideB, utility(resB, pwrB, sideB)


# Number of design points to generate
N = 50000

resA, pwrA, sideA, utlA = design_A_objective_pts(N)
resB, pwrB, sideB, utlB = design_B_objective_pts(N)

res_pts = np.sort(np.hstack([resA, resB]))
power_pts = np.sort(np.hstack([pwrA, pwrB]))
side_pts = np.sort(np.hstack([sideA, sideB]))


# Graph individual utility curves
fig1 = go.Figure(
    go.Scatter(x=res_pts, y=res_qlty(res_pts), mode='lines'),
    layout_title=f'Resolution Utility',
    layout_xaxis_title='Print Head Resolution [in]',
    layout_yaxis_title='Unscaled Utility Score'
)
fig2 = go.Figure(go.Scatter(x=power_pts, y=power_qlty(power_pts), mode='lines'),
    layout_title=f'Power Consumption Utility',
    layout_xaxis_title='Power Consumption [W]',
    layout_yaxis_title='Unscaled Utility Score'
)
fig3 = go.Figure(go.Scatter(x=side_pts, y=side_qlty(side_pts), mode='lines'),
    layout_title=f'Side Length Utility',
    layout_xaxis_title='Side Length [in]',
    layout_yaxis_title='Unscaled Utility Score'
)

fig1.show()
fig2.show()
fig3.show()

# Define independent axis domains for response surface
axis_res = 1000
res_axis = np.linspace(min(res_pts), max(res_pts), axis_res)
pwr_axis = np.linspace(min(power_pts), max(power_pts), axis_res)
side_axis = np.linspace(min(side_pts), max(side_pts), axis_res)


# Plot response surface pairwise
# Resolution and Power
X1, Y1 = np.meshgrid(res_axis, pwr_axis)
utility1 = lambda res_axis, pwr_axis: \
    (3 * scale(res_qlty(res_axis)) + 1 * scale(power_qlty(pwr_axis))) / (3 + 1)
Z1 = utility1(X1, Y1)
figSurf1 = go.Figure(
    go.Surface(x=X1, y=Y1, z=Z1),
    layout_title='Utility Response Surface<br>Projection: Resolution-Power Consumption',
    layout_scene_xaxis_title='Print Head<br>Resolution [in]',
    layout_scene_yaxis_title='Power<br>Consumption [W]',
    layout_scene_zaxis_title='Utility score'
)

# Resolution and Side length
X2, Y2 = np.meshgrid(res_axis, side_axis)
utility2 = lambda res_axis, side_axis: \
    (3 * scale(res_qlty(res_axis)) + 2 * scale(side_qlty(side_axis))) / (3 + 2)
Z2 = utility2(X2, Y2)
figSurf2 = go.Figure(
    go.Surface(x=X2, y=Y2, z=Z2),
    layout_title='Utility Response Surface<br>Projection: Resolution-Side Length',
    layout_scene_xaxis_title='Print Head<br>Resolution [in]',
    layout_scene_yaxis_title='Side Length [in]',
    layout_scene_zaxis_title='Utility score'
)

# Power and Side length
X3, Y3 = np.meshgrid(pwr_axis, side_axis)
utility3 = lambda pwr_axis, side_axis: \
    (1 * scale(power_qlty(pwr_axis)) + 2 * scale(side_qlty(side_axis))) / (1 + 2)
Z3 = utility3(X3, Y3)
figSurf3 = go.Figure(
    go.Surface(x=X3, y=Y3, z=Z3),
    layout_title='Utility Response Surface<br>Projection: Power Consumption-Side Length',
    layout_scene_xaxis_title='Power<br>Consumption [W]',
    layout_scene_yaxis_title='Side Length [in]',
    layout_scene_zaxis_title='Utility score'
)

# Show response surface
figSurf1.show()
figSurf2.show()
figSurf3.show()

# 3D Quality space scatter mapped from form space points
dfA = pd.DataFrame(
    np.array([resA, pwrA, sideA, utlA],).T,
    columns=['Resolution', 'Power Consumption', 'Side Length', 'Utility']
)

dfB = pd.DataFrame(
    np.array([resB, pwrB, sideB, utlB],).T,
    columns=['Resolution', 'Power Consumption', 'Side Length', 'Utility']
)

figSpaceA = px.scatter_3d(
    dfA,
    x='Resolution', y='Power Consumption', z='Side Length',
    color='Utility')
figSpaceA.update_traces(
    marker_size=5,
    marker_line_color='white',
    marker_line_width=2)

figSpaceB = px.scatter_3d(
    dfB,
    x='Resolution', y='Power Consumption', z='Side Length',
    color='Utility')
figSpaceB.update_traces(
    marker_size=5,
    marker_line_color='white',
    marker_line_width=2)

figSpaceA.show()
figSpaceB.show()

# Combined objective space
dfA['Design'] = 'A'
dfB['Design'] = 'B'
df = pd.concat([dfA, dfB], axis=0, ignore_index=True)

figSpace = px.scatter_3d(
    df,
    x='Resolution', y='Power Consumption', z='Side Length',
    color='Design')
figSpace.update_traces(
    marker_size=4,
    marker_line_color='DarkSlateGrey',
    marker_line_width=1)

figSpace.show()
