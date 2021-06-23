from ProblemModule import Design
from ProblemModule.utils import pairplot_overlay, solution_space_similarity, solution_space_overlay

def toy_design(num_pts):   
    reqs_file_A = '/root/ThesisCode/toy_design_files/toy_reqs.json'
    vars_file_A = '/root/ThesisCode/toy_design_files/toy_design_vars.json'
    func_file_A = "/root/ThesisCode/toy_design_files/toy_funcs.json"
    criteria_file_A = "/root/ThesisCode/3DP_criteria.json"

    reqs_file_B = '/root/ThesisCode/toy_design_files/toy_reqs2.json'
    vars_file_B = '/root/ThesisCode/toy_design_files/toy_design_vars2.json'
    func_file_B = "/root/ThesisCode/toy_design_files/toy_funcs2.json"
    criteria_file_B = "/root/ThesisCode/3DP_criteria2.json"

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
    # toy_designA.plot_problem()
    # toy_designA.plot_solutions(show_fails=False)
    # pairplot_overlay(dfA, dfB)
    solution_space_overlay(toy_designA, toy_designB, num_samples=10_000)


def example_design(num_pts):
    reqs_file_A = '/root/ThesisCode/example_design_files/example_design_reqs.json'
    vars_file_A = '/root/ThesisCode/example_design_files/example_design_design_vars.json'
    func_file_A = "/root/ThesisCode/example_design_files/example_design_funcs.json"
    # criteria_file_A = "/root/ThesisCode/example_design_files/example_design_criteria.json"

    reqs_file_B = '/root/ThesisCode/example_design_files/example_design_reqs2.json'
    vars_file_B = '/root/ThesisCode/example_design_files/example_design_design_vars.json'
    func_file_B = "/root/ThesisCode/example_design_files/example_design_funcs.json"
    # criteria_file_B = "/root/ThesisCode/example_design_files/example_design_criteria.json"

    # Generate Design A
    example_designA = Design()
    example_designA.load_requirements_from_json(reqs_file_A)
    example_designA.append_variables_from_json(vars_file_A)
    example_designA.set_map_from_json(func_file_A)
    # example_designA.build_constraint_space()
    example_designA.build_form_space(N=num_pts)

    # Generate Design B
    example_designB = Design()
    example_designB.load_requirements_from_json(reqs_file_B)
    example_designB.append_variables_from_json(vars_file_B)
    example_designB.set_map_from_json(func_file_B)
    # example_designB.build_constraint_space()
    example_designB.build_form_space(N=num_pts)

    # Solution dataframes
    solA = example_designA.form_space.solution_points
    solB = example_designB.form_space.solution_points

    # print(solution_space_similarity(example_designA, example_designB, num_samples=num_pts))

    # plots
    # example_designA.plot_problem()
    # example_designA.plot_solutions(show_fails=False)
    # pairplot_overlay(dfA, dfB)
    solution_space_overlay(example_designA, example_designB, num_samples=num_pts)

num_pts = 1_000_000

print("#"*45)

# toy_design(num_pts)
example_design(num_pts)

print("#"*45)
