import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.express.colors as colors
import string
from time import perf_counter

start=perf_counter()

FILL_COLORS = colors.qualitative.Plotly

scale = lambda X: (X - X.min()) / (X.max() - X.min())


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


# def draw_vertical_line(x_loc, y_range, **kwargs):
#     return go.Scatter(x=[x_loc, x_loc], y=y_range, mode='lines', **kwargs)


# def draw_horizontal_line(y_loc, x_range, **kwargs):
#     return go.Scatter(x=x_range, y=[y_loc, y_loc], mode='lines', **kwargs)


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


# def get_pad(arr):
#     return 0.2 * np.diff(arr) if max(np.abs(arr)) < 1e29 else max(0.2 * min(np.abs(arr)), 5)


# def range_overlays(self, x_range, y_range, **kwargs):
    # x_root = np.nan_to_num(np.product(x_range)) < 0
    # y_root = np.nan_to_num(np.product(y_range)) < 0

    # x_tickvals = [-1, 1] if not x_root else [-1, 0, 1]
    # x_ticktext = [str(x_range[0]), str(x_range[1])] if not x_root else [str(x_range[0]), '0', str(x_range[1])]

    # y_tickvals = [-1, 1] if not y_root else [-1, 0, 1]
    # y_ticktext = [str(y_range[0]), str(y_range[1])] if not y_root else [str(y_range[0]), '0', str(y_range[1])]
    
    # x_lo = -1 if -np.inf in x_range else -1.2
    # x_hi = 1 if np.inf in x_range else 1.2
    # x_limits = [x_lo, x_hi]

    # y_lo = -1 if -np.inf in y_range else -1.2
    # y_hi = 1 if np.inf in y_range else 1.2
    # y_limits = [y_lo, y_hi]

    # plot_kws = dict(
    #     xaxis=dict(
    #         tickmode='array',
    #         tickvals = x_tickvals,
    #         ticktext=x_ticktext
    #     ),
    #     yaxis=dict(
    #         tickmode='array',
    #         tickvals = y_tickvals,
    #         ticktext=y_ticktext
    #     )
    # )

    # box_data = [
    #     draw_vertical_line(-1, y_limits, line=dict(color='blue', width=0.1), showlegend=False),
    #     draw_vertical_line(1, y_limits, line=dict(color='blue', width=0.1), fill='tonexty', showlegend=False),
    #     draw_horizontal_line(-1, x_limits, line=dict(color='red', width=0.1), showlegend=False),
    #     draw_horizontal_line(1, x_limits, line=dict(color='red', width=0.1), fill='tonextx', showlegend=False)
    # ]

    # for line in box_data:
    #     self.add_trace(line)
    # self.update_layout(**plot_kws)
    # self.update_xaxes(range=x_limits)
    # self.update_yaxes(range=y_limits)


def problem_space_overlays(x_range_list, y_range_list, **kwargs):
    '''
    Uses a set of x and y intervals to draw filled boxes in a 2D scatter plot. Allows for visualization of the overlap
    between problem spaces.
    '''
    global FILL_COLORS

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

    return traces, x_axis_params, y_axis_params, X_finite, Y_finite


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
        traces, x_axis_params, y_axis_params, _, _ = problem_space_overlays(
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


def points_2_problem_space(self, X, range_list, **kwargs):
    '''
    Extension method for go.Figure

    Creates a problem space overlay for every pair of axes in the problem space and arranges subplots into upper
    triangular subplot matrix.
    '''

    # grid_dim = len(range_list) - 1
    # self.set_subplots(grid_dim, grid_dim, vertical_spacing=0.01, horizontal_spacing=0.03)

    # subplot_idxs = [(i + 1, j + 1) for i in range(grid_dim) for j in range(i, grid_dim)]

    # axis_labels = kwargs.get('axis_labels', [f'Axis {i}' for i in range(grid_dim + 1)])
    # kwargs = {key: value for key, value in kwargs.items() if key not in ['axis_labels']}

    # for i, j in subplot_idxs:
    #     x_idx = j
    #     y_idx = i - 1

    #     self.add_trace(draw_box(range_list[x_idx], range_list[y_idx], **kwargs), row=i, col=j)
    #     self.add_trace(go.Scatter(x=X[x_idx], y=X[y_idx], mode='markers', **kwargs), row=i, col=j)


    pass

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
        traces, x_axis_params, y_axis_params, X_scaled, Y_scaled = problem_space_overlays(
            x_ranges,
            y_ranges,
            showlegend=idx == 0,  # Only show each space in legend once
            **kwargs)

        # Add each trace to appropriate subplot and specify axis params for subplot
        for trace in traces:
            self.add_trace(trace, row=i, col=j)

        x_min, x_max = X_scaled.min(), X_scaled.max()
        y_min, y_max = Y_scaled.min(), Y_scaled.max()

        x_uns = X[j]
        y_uns = X[i-1]

        x_uns[x_uns==np.inf] = x_max
        y_uns[y_uns==np.inf] = y_max
        x_uns[x_uns==-np.inf] = x_min
        y_uns[y_uns==-np.inf] = y_min

        x = (x_uns - x_min) / (x_max - x_min)
        y = (y_uns - y_min) / (y_max - y_min)

        self.add_trace(
            go.Scatter(
                x=x, y=y,
                mode='markers', marker_color='red', marker_size=4,
                showlegend=False, **kwargs),
            row=i, col=j)

        self.update_xaxes(
            x_axis_params, row=i, col=j,  # Add axis params from problem_space_overlays method
            ticks="outside" if i == 1 else None,  # Only place ticks on certain subplots
            showticklabels=i == 1, tickangle=-45,  # Only place axis label on certain subplots, rotate tickval
            title=axis_labels[j] if i == 1 else None,  # Only show axis title on certain subplots
            tickfont=dict(size=11),  # Specify tickval font parameters
            side = 'top' if i == 1 else None,  # Specify which edge of subplot to place axis labels / title
            matches = 'x'
        )

        self.update_yaxes(
            y_axis_params, row=i, col=j,  # Add axis params from problem_space_overlays method
            ticks="outside" if j == grid_dim else None,  # Only place ticks on certain subplots
            showticklabels=j == grid_dim, tickangle=0,  # Only place axis label on certain subplots, rotate tickval
            title=axis_labels[i - 1] if j == grid_dim else None,  # Only show axis title on certain subplots
            tickfont=dict(size=11),  # Specify tickval font parameters
            side = 'right' if j == grid_dim else None,  # Specify which edge of subplot to place axis labels / title
            matches = 'y'
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

if __name__ == "__main__":

    # Add extensions to imported objects
    go.Figure.problem_space_grid = problem_space_grid
    go.Figure.points_2_problem_space = points_2_problem_space

    # A_rg1 = [-2, np.inf]
    # B_rg1 = [5, 30]
    # C_rg1 = [-10, 10]
    # D_rg1 = [-np.inf, 7]

    # A_rg2 = [-5, np.inf]
    # B_rg2 = [0, 25]
    # C_rg2 = [-15, 15]
    # D_rg2 = [-np.inf, 26]

    # ranges = [A_rg1, B_rg1, C_rg1, D_rg1]
    # ranges = [[A_rg1, A_rg2], [B_rg1, B_rg2], [C_rg1, C_rg2], [D_rg1, D_rg2]]

    rgA = [
        [-0.005, 0.005],
        [-0.001, 0.001],
        [-0.005, 0.005],
        [8.0, 36],
        [8.0, 36],
        [16.0, 100],
        [0.0, 0.001]
    ]

    rgB = [
        [-0.005, 0.005],
        [-0.001, 0.001],
        [-0.005, 0.005],
        [8.0, 36],
        [8.0, 36],
        [8.0, 100],
        [0.0, 0.001]
    ]

    ranges = list(zip(rgA, rgB))
    axis_labels = ['dx', 'dG', 'dy', 'Dx', 'Dy', 'v', 'res']
    # axis_labels = ['Frame<br>Deflection<br>X', 'Gantry<br>Deflection', 'Frame<br>Deflection<br>Y', 'X-Travel', 'Y-Travel', 'Max<br>Printhead<br>Velocity', 'Print<br>Head<br>Resolution']

    N = 50
    # ptsA = choose_ndim_point(rgA, N)
    ptsB = choose_ndim_point(rgB, N)
    # points = np.hstack([ptsA, ptsB])
    # in_both = np.logical_and(isinspace(points.T, rgA), isinspace(points.T, rgB))

    fig = go.Figure()
    fig.problem_space_grid(ranges, axis_labels=axis_labels)
    # fig.points_2_problem_space(ptsB, rgB, axis_labels=axis_labels)

    fig.show()

    print(round(perf_counter() - start, 2))
