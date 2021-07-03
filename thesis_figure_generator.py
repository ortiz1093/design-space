from ProblemModule import Design
from ProblemModule.utils import pairplot_overlay, solution_space_similarity, solution_space_overlay
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
from sympy import sympify, lambdify


# Function for MinMax scaling data
scale = lambda X: (X - X.min()) / (X.max() - X.min())

# Specify utility curve for each criterion
res_qlty = lambda x: 1 / x
power_qlty = lambda x: np.exp(-x)
side_qlty = lambda x: x

# Utility function
utility = lambda res_axis, pwr_axis, side_axis: \
    (3 * scale(res_qlty(res_axis)) + 1 * scale(power_qlty(pwr_axis)) + 2 * scale(side_qlty(side_axis))) / (3 + 1 + 2)


def str2function(str_expression):
    sym_exp = sympify(str_expression)
    inputs = sorted([str(symbol) for symbol in sym_exp.free_symbols])

    return lambdify(inputs, sym_exp, 'numpy')


def get_objective_points(formA_dict, solnA, formB_dict, solnB):
    motor_V_opts = np.array([3.5, 4.0, 12.0, 6.0, 24.0, 6.0, 12.0, 3.4, 12.0, 3.06])
    motor_Amp_opts = np.array([1.0, 1.5, 3.6, 1.0, 0.3, 1.3, 0.7, 2.8, 0.8, 0.7])

    ########## Design A ################
    muA = formA_dict['mu'][solnA]
    pA = formA_dict['p'][solnA]
    nA = formA_dict['n'][solnA]
    phiA = [1.8]*len(muA)
    motorA = formA_dict['Motor'][solnA].astype(int) - 1
    motor_VA = motor_V_opts[motorA]
    motor_AmpA = motor_Amp_opts[motorA]
    sA = formA_dict['s'][solnA]

    # Get design criteria
    criteria_fileA = 'example_design_files/DesignA_coreXY/design_criteria_A.json'
    criteria_dictA = json.load(open(criteria_fileA, 'r'))

    # Convert string expressions into criteria map
    resolutionA = str2function(criteria_dictA['res'])
    power_usedA = str2function(criteria_dictA['pwr'])
    side_lenA = str2function(criteria_dictA['s'])

    # Objective space points    
    resA = resolutionA(muA, nA, pA, phiA)
    pwrA = power_usedA(motor_AmpA, motor_VA)
    sideA = side_lenA(sA)
    utlA = utility(resA, pwrA, sideA)

    ########## Design B ################
    screw_p_opts = np.array([0.012, 0.049, 0.159, 0.025, 0.125, 0.200, 0.039, 0.196, 0.049, 0.197])
    screw_nu_opts = np.array([21, 89, 86, 21, 84, 84, 79, 85, 86, 88])

    muB = formB_dict['mu'][solnB]
    phiB = [1.8]*len(muB)

    screwB = formB_dict['Screw'][solnB].astype(int) - 1
    screw_pB = screw_p_opts[screwB]
    screw_nuB = screw_nu_opts[screwB]


    motorB = formB_dict['Motor'][solnB].astype(int) - 1
    motor_VB = motor_V_opts[motorB]
    motor_AmpB = motor_Amp_opts[motorB]
    sB = formB_dict['s'][solnB]

    # Get design criteria
    criteria_fileB = 'example_design_files/DesignB_leadScrew/design_criteria_B.json'
    criteria_dictB = json.load(open(criteria_fileB, 'r'))

    # Convert string expressions into criteria map
    resolutionB = str2function(criteria_dictB['res'])
    power_usedB = str2function(criteria_dictB['pwr'])
    side_lenB = str2function(criteria_dictB['s'])

    # Objective space points    
    resB = resolutionB(muB, screw_pB, phiB)
    pwrB = power_usedB(motor_AmpB, motor_VB, screw_nuB)
    sideB = side_lenB(sB)
    utlB = utility(resB, pwrB, sideB)

    return resA, pwrA, sideA, utlA, resB, pwrB, sideB, utlB


def objective_space(resA, pwrA, sideA, utlA, resB, pwrB, sideB, utlB):
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
        color='Utility',
        title="Design A<br>Quality Space<br>Utility Gradient Applied")
    figSpaceA.update_traces(
        marker_size=5,
        marker_line_color='DarkSlateGrey',
        marker_line_width=0.25)

    figSpaceB = px.scatter_3d(
        dfB,
        x='Resolution', y='Power Consumption', z='Side Length',
        color='Utility',
        title="Design B<br>Quality Space<br>Utility Gradient Applied")
    figSpaceB.update_traces(
        marker_size=5,
        marker_line_color='DarkSlateGrey',
        marker_line_width=0.25)

    figSpaceA.show()
    figSpaceB.show()

    # Combined objective space
    dfA['Design'] = 'A'
    dfB['Design'] = 'B'
    df = pd.concat([dfA, dfB], axis=0, ignore_index=True)

    figSpace = px.scatter_3d(
        df,
        x='Resolution', y='Power Consumption', z='Side Length',
        color='Design',
        title="Designs A & B<br>Quality Space Comparison<br>Utility Gradient Not Applied")
    figSpace.update_traces(
        marker_size=4,
        marker_line_color='DarkSlateGrey',
        marker_line_width=0.25)

    figSpace.show()


def conformity(form_dict, soln_mask, outlier):
    N_plots = len(form_dict.keys()) - 1
    N = sum(soln_mask)

    form_pts = np.array(list(form_dict.values())).astype(float)
    pts = form_pts[:, soln_mask]

    x = np.array(list(outlier.values())).astype(float).flatten()

    dists = np.linalg.norm(pts.T - np.tile(x, [N, 1]), axis=1)
    NN_idx = np.argmin(dists)
    
    axis_labels = list(form_dict.keys())
    
    if N_plots == 1:
        fig = go.Figure(
            # layout_xaxis=dict(
            #     title=axis_labels[0]
            # ),
            # layout_yaxis=dict(
            #     title=axis_labels[1],
            # ),
            layout_height=800,
            layout_width=600
        )
        fig.add_trace(
            go.Scatter(
                x=pts[0],
                y=pts[1],
                showlegend=False,
                mode='markers', marker_color='rgb(47, 138, 196)',
                marker_line_width=0.5, marker_line_color='white'
        ))

        NN = pts.T[NN_idx]
        NN_line = list(zip(x, NN))

        fig.add_trace(
            go.Scatter(
                x=NN_line[0], y=NN_line[1],
                showlegend=False,
                mode='markers + lines',
                line_color='rgb(237, 100, 90)', line_width=3,
                marker_color='rgb(118, 78, 159)', marker_size=10
        ))
        fig.add_trace(
            go.Scatter(
                x=[x[0]], y=[x[1]],
                showlegend=False,
                mode='markers',
                marker_color='red', marker_size=10
        ))
        fig.update_layout(
            xaxis_title=r'$\text{Radius} \; [\text{m}]$',
            yaxis_title=r'$\rho [\text{kg/m}^3]$',
            font=dict(size=18)
        )
        fig.show()

    else:
        plot_idxs = [(i + 1, j + 1) for i in range(N_plots) for j in range(i, N_plots)]
        fig = make_subplots(rows=N_plots, cols=N_plots, vertical_spacing=0.01, horizontal_spacing=0.03)
        for i, j in plot_idxs:
            fig.add_trace(
                go.Scatter(
                    x=pts[j], y=pts[i-1],
                    showlegend=False,
                    mode='markers', marker_color='rgb(47, 138, 196)',
                    marker_line_width=0.5, marker_line_color='white'),
                row=i, col=j
            )

            projected_pts = np.array([pts[j], pts[i-1]])
            x_proj = np.array([x[j], x[i-1]])
            NN = projected_pts.T[NN_idx]
            NN_line = list(zip(x_proj, NN))

            fig.add_trace(
                go.Scatter(
                    x=NN_line[0], y=NN_line[1],
                    showlegend=False,
                    mode='markers + lines',
                    line_color='rgb(237, 100, 90)', line_width=3,
                    marker_color='rgb(118, 78, 159)', marker_size=8),
                row=i, col=j
            )

            fig.add_trace(
                go.Scatter(
                    x=[x_proj[0]], y=[x_proj[1]],
                    showlegend=False,
                    mode='markers',
                    marker_color='red', marker_size=8),
                row=i, col=j
            )

            x_flag = i==1
            y_flag = j==N_plots

            fig.update_xaxes(
                row=i, col=j,
                ticks="outside",  # Only place ticks on certain subplots
                visible=x_flag, tickangle=-45,  # Only place axis label on certain subplots, rotate tickval
                title=axis_labels[j],  # Only show axis title on certain subplots
                tickfont=dict(size=11),  # Specify tickval font parameters
                side = 'top',  # Specify which edge of subplot to place axis labels / title
            )
            
            fig.update_yaxes(
                row=i, col=j,
                ticks="outside",  # Only place ticks on certain subplots
                visible=y_flag, tickangle=0,  # Only place axis label on certain subplots, rotate tickval
                title=axis_labels[i - 1],  # Only show axis title on certain subplots
                tickfont=dict(size=11),  # Specify tickval font parameters
                side = 'right',  # Specify which edge of subplot to place axis labels / title
            )
        fig.show()


def toy_design(num_pts):   
    reqs_file_A = '/root/ThesisCode/toy_design_files/toy_reqs.json'
    vars_file_A = '/root/ThesisCode/toy_design_files/toy_design_vars.json'
    func_file_A = "/root/ThesisCode/toy_design_files/toy_funcs.json"

    reqs_file_B = '/root/ThesisCode/toy_design_files/toy_reqs2.json'
    vars_file_B = '/root/ThesisCode/toy_design_files/toy_design_vars2.json'
    func_file_B = "/root/ThesisCode/toy_design_files/toy_funcs2.json"

    # Generate Design A
    toy_designA = Design()
    toy_designA.load_requirements_from_json(reqs_file_A)
    toy_designA.append_variables_from_json(vars_file_A)
    toy_designA.set_map_from_json(func_file_A)
    toy_designA.build_constraint_space()
    toy_designA.build_form_space(N=num_pts)

    # Generate Design B
    toy_designB = Design()
    toy_designB.load_requirements_from_json(reqs_file_B)
    toy_designB.append_variables_from_json(vars_file_B)
    toy_designB.set_map_from_json(func_file_B)
    toy_designB.build_constraint_space()
    toy_designB.build_form_space(N=num_pts)

    # Solution dataframes
    solA = toy_designA.form_space.solution_points
    solB = toy_designB.form_space.solution_points

    dfA = toy_designA.form_space.points_df[solA]
    dfB = toy_designB.form_space.points_df[solB]

    # print(solution_space_similarity(toy_designA, toy_designB, num_samples=1_000_000))

    # plots
    toy_designA.plot_problem()
    # toy_designA.plot_solutions(show_fails=False)
    # pairplot_overlay(dfA, dfB)
    # solution_space_overlay(toy_designA, toy_designB, num_samples=num_pts)
    # x = dict(
    #     r=1.25,
    #     rho=980
    # )
    # conformity(toy_designA.form_points, solA, x)


def example_design(num_pts):
    reqs_file_A = 'example_design_files/example_design_reqs.json'
    vars_file_A = 'example_design_files/DesignA_coreXY/design_vars_A.json'
    func_file_A = 'example_design_files/DesignA_coreXY/design_funcs_A.json'
    # criteria_file_A = "/root/ThesisCode/example_design_files/example_design_criteria.json"

    reqs_file_B = 'example_design_files/example_design_reqs.json'
    vars_file_B = 'example_design_files/DesignB_leadScrew/design_vars_B.json'
    func_file_B = 'example_design_files/DesignB_leadScrew/design_funcs_B.json'
    # criteria_file_B = "/root/ThesisCode/example_design_files/example_design_criteria.json"

    # Generate Design A
    designA = Design()
    designA.load_requirements_from_json(reqs_file_A)
    designA.append_variables_from_json(vars_file_A)
    designA.set_map_from_json(func_file_A)
    # designA.build_constraint_space()
    designA.build_form_space(N=num_pts)

    # Generate Design B
    designB = Design()
    designB.load_requirements_from_json(reqs_file_B)
    designB.append_variables_from_json(vars_file_B)
    designB.set_map_from_json(func_file_B)
    # designB.build_constraint_space()
    designB.build_form_space(N=num_pts)

    rgsA = [req.values for req in designA.requirement_set]
    rgsB = [req.values for req in designB.requirement_set]
    syms = [req.symbol for req in designB.requirement_set]
    # print(syms)
    # quit()

    # Solution dataframes
    solA = designA.form_space.solution_points
    solB = designB.form_space.solution_points

    # resA, pwrA, sideA, utlA, resB, pwrB, sideB, utlB = get_objective_points(designA.form_points, solA, designB.form_points, solB)
    # objective_space(resA, pwrA, sideA, utlA, resB, pwrB, sideB, utlB)

    x = dict(
        s=26,
        q=9,
        t=0.21,
        Y=12.5e6,
        G=40,
        w=1.25,
        h=0.5,
        t_f=0.02,
        t_w=0.175,
        mu=0.125,
        p=3.25,
        n=35,
        Motor=2
    )
    conformity(designA.form_points, solA, x)

    # print(solution_space_similarity(designA, designB, num_samples=num_pts))

    # plots
    # designA.plot_problem()
    # designA.plot_solutions(show_fails=False)
    # pairplot_overlay(solA, solB)
    # solution_space_overlay(designA, designB, num_samples=num_pts)

num_pts = 50_000

print("#"*45)

toy_design(num_pts)
# example_design(num_pts)

print("#"*45)
