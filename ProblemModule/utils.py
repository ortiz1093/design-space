import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from plotly.subplots import make_subplots
import plotly.express.colors as colors
import string


def continuous_continuous_space(rangeA, rangeB, color='plum', name=None,
                                showlegend=True):
    """
    Generate a plot showing a shaded continuous region of 2D space (rectangle).
        params:
            rangeA: upper and lower bounds of horizontal axis
            rangeB: upper and lower bounds of vertical axis
            color (optional): color of shaded region
            name (optional): name of shaded region to show in legend
            showlegend (optional): whether to represent trace in plot legend
        return:
            2D plotly scatter trace
    """
    x0, x1 = min(rangeA), min(max(rangeA), 1e300)
    y0, y1 = min(rangeB), min(max(rangeB), 1e300)

    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode='markers',
        fill='toself',
        fillcolor=color,
        line=dict(color=color),
        name=name,
        legendgroup="group",
        showlegend=showlegend
    )


def dep_continuous_continuous_space(rangeA, rangeB, color='plum', name=None,
                                    showlegend=True):
    """
    Generate a plot showing a shaded continuous region of 2D space (rectangle).
        params:
            rangeA: upper and lower bounds of horizontal axis
            rangeB: upper and lower bounds of vertical axis
            color (optional): color of shaded region
            name (optional): name of shaded region to show in legend
            showlegend (optional): whether to represent trace in plot legend
        return:
            2D plotly scatter trace
    """
    x0, x1 = min(rangeA), min(max(rangeA), 1e300)
    y0, y1 = min(rangeB), min(max(rangeB), 1e300)

    return go.Scatter(
        x=[x0, x1, x1, x0, x0],
        y=[y0, y0, y1, y1, y0],
        mode='markers',
        fill='toself',
        fillcolor=color,
        line=dict(color=color),
        name=name,
        legendgroup="group",
        showlegend=showlegend
    )


def mask_union(maskA, maskB):
    setA = set(np.argwhere(maskA).flatten())
    setB = set(np.argwhere(maskB).flatten())
    # TODO: modify to return the indices that are True in either, not the truth values themselves
    return setA.union(setB)


def mask_intersection(maskA, maskB):
    setA = set(np.argwhere(maskA).flatten().copy())
    setB = set(np.argwhere(maskB).flatten().copy())
    # TODO: modify to return the indices that are True in both, not the truth values themselves
    return setA.intersection(setB)


def jaccard_index(maskA, maskB):
    numerator = len(mask_intersection(maskA, maskB))
    denominator = len(mask_union(maskA, maskB)) + 1e-300

    return numerator / denominator


def overlap_coefficient(maskA, maskB):
    numerator = len(mask_intersection(maskA, maskB))
    denominator = min(len(maskA[maskA]), len(maskB[maskB])) + 1e-300

    return numerator / denominator


def space_similarity(maskA, maskB):
    oc = overlap_coefficient(maskA, maskB)
    J = jaccard_index(maskA, maskB)

    return np.sqrt(J**2 + oc**2) / np.sqrt(2)


def test_shared_form_points(designA, designB, samples, return_samples=False):
    # Generate common set of points to map
    # samples = designA.generate_samples(num_samples)
    inputs = dict([(item[0], item[1])
                    for item in samples if item[2]])

    # Map point sets for both designs and get solution masks
    pointsA = designA.map.map_points(inputs)
    soln_mask_A = designA.requirement_set.check_compliance(pointsA)

    pointsB = designB.map.map_points(inputs)
    soln_mask_B = designB.requirement_set.check_compliance(pointsB)

    if return_samples:
        return soln_mask_A, soln_mask_B, samples

    return soln_mask_A, soln_mask_B


def solution_space_similarity(designA, designB, **kwargs):
    samples = designA.generate_samples(kwargs['num_samples'])
    soln_mask_A, soln_mask_B = test_shared_form_points(designA, designB, samples=samples, return_samples=False)

    # Run similarity measure on masks
    return space_similarity(soln_mask_A, soln_mask_B)


def pairplot_overlay(df1, df2):
    assert np.all(sorted(df1.columns)==sorted(df2.columns)), "DataFrames must have matching columns"

    labels = [0]*len(df1) + [1]*len(df2)
    df2_sorted = df2[df1.columns]
    df = pd.concat([df1, df2_sorted], axis=0, ignore_index=True)
    df['labels'] = labels

    sns.pairplot(df, hue='labels', diag_kind='kde', corner=True, aspect=1, height=1)
    plt.show()


def solution_space_overlay(designA, designB, **kwargs):
    axisTitleFontSize = 28
    layoutFontSize = 18

    samples = designA.generate_samples(kwargs['num_samples'])
    mask_A, mask_B, samples = test_shared_form_points(designA, designB, samples=samples, return_samples=True)
    mask_Both = np.logical_and(mask_A, mask_B)
    soln_idxs = np.logical_or(mask_A, mask_B)

    col_names = [var[0] for var in samples if var[2]]
    sample_data = np.vstack(([var[1] for var in samples if isinstance(var, tuple) and var[2]])).T

    df = pd.DataFrame(sample_data, columns=col_names)

    labels = np.full(len(df), np.nan)
    labels[mask_A] = 0
    labels[mask_B] = 1
    labels[mask_Both] = 2

    df['labels'] = labels
    hue = 'labels'

    import plotly.express as px

    plot = px.scatter_matrix(
            df[labels==0],
            dimensions=df.columns[:-1])
    plot2 = px.scatter_matrix(
            df[labels==1],
            dimensions=df.columns[:-1])
    plot3 = px.scatter_matrix(
            df[labels==2],
            dimensions=df.columns[:-1])
    plot.add_trace(plot2.data[0])
    plot.add_trace(plot3.data[0])
    plot.update_traces(
        diagonal_visible=False,
        showupperhalf=False,
        marker=dict(
            color=labels,
            size=4, opacity=1.0,
            showscale=False, # colors encode categorical variables
            line_color='whitesmoke', line_width=0.5))
    plot.show()


def dep_solution_space_overlay(designA, designB, **kwargs):
    axisTitleFontSize = 28
    layoutFontSize = 18

    samples = designA.generate_samples(kwargs['num_samples'])
    mask_A, mask_B, samples = test_shared_form_points(designA, designB, samples=samples, return_samples=True)
    mask_Both = np.logical_and(mask_A, mask_B)
    # soln_idxs = np.logical_or(mask_A, mask_B)

    col_names = [var[0] for var in samples if var[2]]
    sample_data = np.vstack(([var[1] for var in samples if isinstance(var, tuple) and var[2]])).T

    df = pd.DataFrame(sample_data, columns=col_names)

    labels = np.full(len(df), np.nan)
    labels[mask_A] = 0
    labels[mask_B] = 1
    labels[mask_Both] = 2

    # labels = np.full(len(df), '', dtype=np.dtype('U4'))
    # labels[mask_A] = 'A'
    # labels[mask_B] = 'B'
    # labels[mask_Both] = 'Both'

    df['labels'] = labels
    hue = 'labels'

    if len(col_names) == 2:
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df.iloc[mask_A, 0], y=df.iloc[mask_A, 1],
                    mode='markers', marker_color='blue', name='A'
                ),
                go.Scatter(
                    x=df.iloc[mask_B, 0], y=df.iloc[mask_B, 1],
                    mode='markers', marker_color='red', name='B'
                ),
                go.Scatter(
                    x=df.iloc[mask_Both, 0], y=df.iloc[mask_Both, 1],
                    mode='markers', marker_color='purple', name='Both'
                ),
            ],
            layout_xaxis_title=dict(
                text=col_names[0],
                font_size=axisTitleFontSize
            ),
            layout_yaxis_title=dict(
                text=col_names[1],
                font_size=axisTitleFontSize,
            ),
            layout_font=dict(size=layoutFontSize),
            layout_scene_aspectmode='manual',
            layout_scene_aspectratio=dict(x=0, y=0, z=0)
        )
        fig.show()
    else:
        def onclick(event):
            axes = event.inaxes
            axis_names = df.columns
            # x_name = axis_names[axes.colNum]
            # y_name = axis_names[axes.rowNum]
            x_name = axis_names[axes.get_subplotspec().colspan.start]
            y_name = axis_names[axes.get_subplotspec().rowspan.start]
            x = df[x_name]
            y = df[y_name]
            hu = df[hue] if hue else None
            plt.figure()
            clk_ax = sns.scatterplot(x=x, y=y, hue=hu, palette='deep')
            clk_ax.xaxis.xlabel = x_name
            clk_ax.xaxis.ylabel = y_name
            plt.show()
            # print(event.inaxes)
            # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #     ('double' if event.dblclick else 'single', event.button,
            #     event.x, event.y, event.xdata, event.ydata))

        # df.labels.map({0: 'A', 1: 'B', 2: 'Both'})
        df.labels = df.labels.replace({0: 'A', 1: 'B', 2: 'Both'})
        plot = sns.pairplot(
            df, hue=hue, diag_kind='kde', corner=True, aspect=1, height=0.9, palette='deep',
            plot_kws=dict(s=8), diag_kws=dict(visible=False))
        
        # plot = sns.PairGrid(df, hue=hue, corner=True, height=1, aspect=1)
        # plot.map_lower(sns.scatterplot)
        # plot.map_diag(sns.kdeplot)
        # plot.add_legend(labels=['A','B','Both'])


        # handles = plot._legend_data.values()
        # plot.fig.legend(handles=handles, labels=['A','B','Both'])

        plot.fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

############################# Problem Space Overlays #####################################

def choose_ndim_point(axis_intervals, N):
    intervals = np.nan_to_num(axis_intervals)
    return np.array(
        [np.random.uniform(*interval, N) for interval in intervals]
    )


def isinspace(pts, space_intervals):
    results = []
    for pt in pts:
        results.append(
            np.all(
                [min(rg) < val < max(rg) for val, rg in zip(pt, space_intervals)]
            )
        )
    return results


def draw_box(x_range, y_range, **kwargs):
    '''
    Draws a box in a 2D plane given axis intervals
    '''

    # Get box corners from given intervals
    x0, x1 = sorted(x_range)
    y0, y1 = sorted(y_range)

    opacity = kwargs.get('opacity', 0.4)
    kwargs = {key: value for key, value in kwargs.items() if key not in ['opacity']}

    # Create a scatter plot from corner points, remove lines and markers, fill box
    trace =  go.Scatter(
        x=[x0, x0, x1, x1, x0], y=[y0, y1, y1, y0, y0],
        mode='none', fill='toself', opacity=opacity,
        **kwargs)
    
    return trace


def problem_space_overlays(x_range_list, y_range_list, **kwargs):
    '''
    Uses a set of x and y intervals to draw filled boxes in a 2D scatter plot. Allows for visualization of the overlap
    between problem spaces.
    '''
    FILL_COLORS = colors.qualitative.Plotly

    # Convert inputs to ndarrays
    X = np.array(x_range_list)
    Y = np.array(y_range_list)

    # Sets buffer between boxes and edges of plot except when space extends to infinity
    X_pad_lo = 0 if -np.inf in X else 0.2
    X_pad_hi = 0 if np.inf in X else 0.2
    Y_pad_lo = 0 if -np.inf in Y else 0.2
    Y_pad_hi = 0 if np.inf in Y else 0.2

    # Changes inf values to be a multiple of the highest finite value so that the plot scale makes the overlap visible
    inf_pad = 2
    X_finite, Y_finite = X.copy(), Y.copy()
    X_finite[X==np.inf] = inf_pad * np.sort(X[X!=np.inf].flatten())[-1]
    Y_finite[Y==np.inf] = inf_pad * np.sort(Y[Y!=np.inf].flatten())[-1]
    X_finite[X==-np.inf] = -inf_pad * np.sort(X[X!=np.inf].flatten())[-1]
    Y_finite[Y==-np.inf] = -inf_pad * np.sort(Y[Y!=np.inf].flatten())[-1]

    # Scale ranges for nicer plotting, puts everything on interval [0, 1]
    X_scaled = (X_finite - X_finite.min()) / (X_finite.max() - X_finite.min())
    Y_scaled = (Y_finite - Y_finite.min()) / (Y_finite.max() - Y_finite.min())

    # Get color from kwargs if given, else set color
    fillcolor = kwargs.get('fillcolor')  # Get space color if provided in function call
    color_flag = fillcolor is not None  # Flag whether color was provided in function call
    plotly_colors = iter(FILL_COLORS)  # Create an iterable of color options for coloring spaces

    # Get name from kwargs if given, else set name
    name = kwargs.get('name')  # Get space label if provided in function call
    name_flag = name is not None  # Flag whether name was provided in function call
    letters = iter(string.ascii_letters[26:])  # Create an iterable of capital letters for labeling spaces

    # Remove name and fillcolor from kwargs dict and repackage for passing to next function
    kwargs = {key: value for key, value in kwargs.items() if key not in ['name', 'fillcolor']}

    num_boxes = X.shape[0]  # Number of spaces to draw
    traces = []  # Collector for traces

    for i in range(num_boxes):
        # Set name and fillcolor
        fillcolor = fillcolor if color_flag else next(plotly_colors)
        name = name if name_flag else f'Space {next(letters)}'

        # Call to draw_box to create a filled box for each space
        traces.append(
            draw_box(
                X_scaled[i], Y_scaled[i],
                fillcolor=fillcolor,
                name=name,
                **kwargs)
        )
    
    # Uses edges of scaled boxes for tick locations
    x_tickvals = np.sort(X_scaled.flatten())
    y_tickvals = np.sort(Y_scaled.flatten())

    # Relabel ticks with original interval limits (gives appearance that boxes actually extend to original limits)
    x_ticktext = np.sort(X.flatten()).astype(str)
    y_ticktext = np.sort(Y.flatten()).astype(str)

    # Use padding from earlier to create space between boxes and plot edge except when space goes to infinity
    x_range = [X_scaled.min() - X_pad_lo, X_scaled.max() + X_pad_hi]
    y_range = [Y_scaled.min() - Y_pad_lo, Y_scaled.max() + Y_pad_hi]

    # Collect axis settings in dict for easy passing to go.Layout
    x_axis_params = dict(
        range=x_range,
        tickmode='array',
        tickvals=x_tickvals,
        ticktext=x_ticktext
    )
    y_axis_params = dict(
        range=y_range,
        tickmode='array',
        tickvals=y_tickvals,
        ticktext=y_ticktext
    )

    return traces, x_axis_params, y_axis_params


def problem_space_grid(self, range_list, **kwargs):
    '''
    Extension method for go.Figure

    Creates a problem space overlay for every pair of axes in the problem space and arranges subplots into upper
    triangular subplot matrix.
    '''

    # Set number of rows and columns and create subplots
    grid_dim = len(range_list) - 1
    self.set_subplots(grid_dim, grid_dim, vertical_spacing=0.01, horizontal_spacing=0.03)

    # Create list of coordinate pairs for faster looping
    axis_coords = [(i + 1, j + 1) for i in range(grid_dim) for j in range(i, grid_dim)]

    # Get axis labels if provided in method call and remove from kwargs dict
    axis_labels = kwargs.get('axis_labels', [f'Axis {i}' for i in range(grid_dim + 1)])
    kwargs = {key: value for key, value in kwargs.items() if key not in ['axis_labels']}

    for idx, (i, j) in enumerate(axis_coords):

        # If only one range provided, convert to nested list else leave as is (for problem_space_overlays compatibility)
        x_ranges = range_list[j] if isinstance(range_list[j][0], list) else [range_list[j]]
        y_ranges = range_list[i-1] if isinstance(range_list[i-1][0], list) else [range_list[i-1]]

        # get traces and axis parameters for each subplot from problem_space_overlays function
        traces, x_axis_params, y_axis_params = problem_space_overlays(
            x_ranges,
            y_ranges,
            showlegend=idx == 0,  # Only show each space in legend once
            **kwargs)

        # Add each trace to appropriate subplot and specify axis params for subplot
        for trace in traces:
            self.add_trace(trace, row=i, col=j)

        self.update_xaxes(
            x_axis_params, row=i, col=j,  # Add axis params from problem_space_overlays method
            ticks="outside" if i == 1 else None,  # Only place ticks on certain subplots
            showticklabels=i == 1, tickangle=-45,  # Only place axis label on certain subplots, rotate tickval
            title=axis_labels[j] if i == 1 else None,  # Only show axis title on certain subplots
            tickfont=dict(size=11),  # Specify tickval font parameters
            side = 'top' if i == 1 else None,  # Specify which edge of subplot to place axis labels / title
            # title_standoff = 40, automargin=False
        )
        # self.update_xaxes(
        #     side='bottom',
        #     # showticklabels=False,
        #     title=axis_labels[j],
        #     overlaying=f'x{j}'
        # )

        self.update_yaxes(
            y_axis_params, row=i, col=j,  # Add axis params from problem_space_overlays method
            ticks="outside" if j == grid_dim else None,  # Only place ticks on certain subplots
            showticklabels=j == grid_dim, tickangle=0,  # Only place axis label on certain subplots, rotate tickval
            title=axis_labels[i - 1] if j == grid_dim else None,  # Only show axis title on certain subplots
            tickfont=dict(size=11),  # Specify tickval font parameters
            side = 'right' if j == grid_dim else None,  # Specify which edge of subplot to place axis labels / title
            # title_standoff = 500, automargin=False
        )

    # Specify common layout paramters
    self.update_layout(
        legend=dict(
            yanchor="bottom",
            y=0.3,
            xanchor="left",
            x=0.3
        )
    )


##################################################################################################
##################################################################################################
if __name__ == "__main__":
    from numpy.random import default_rng

    rng = default_rng(42)

    bools = [True, False]
    
    A = rng.choice(bools, 10)
    B = rng.choice(bools, 10)

    J = jaccard_index(A, B)
    oc = overlap_coefficient(A, B)

    ss = solution_space_similarity(A, B)

    # Not measuring the right thing, need to know which points are in both or either, not the bool values of the points
    print(f"Set A: {A}")
    print(f"Set B: {B}")
    print(f"Union: {mask_union(A, B)}")
    print(f"Intersection{mask_intersection(A, B)}")
    print(f"Jaccard: {J}")
    print(f"Overlap: {oc}")
    print(f"Similarity: {ss}")
    pass
