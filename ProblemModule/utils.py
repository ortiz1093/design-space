import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
    samples = designA.generate_samples(kwargs['num_samples'])
    mask_A, mask_B, samples = test_shared_form_points(designA, designB, samples=samples, return_samples=True)
    mask_Both = np.logical_and(mask_A, mask_B)

    col_names = [var[0] for var in samples if var[2]]
    sample_data = np.vstack(([var[1] for var in samples if isinstance(var, tuple) and var[2]])).T

    df = pd.DataFrame(sample_data, columns=col_names)

    labels = np.full(len(df), np.nan)
    labels[mask_A] = 0
    labels[mask_B] = 1
    labels[mask_Both] = 2

    df['labels'] = labels
    hue = 'labels'

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

    plot = sns.pairplot(df, hue=hue, diag_kind='kde', corner=True, aspect=1, height=0.9, palette='deep', plot_kws=dict(s=8))
    plot.fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


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
