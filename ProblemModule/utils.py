import plotly.graph_objects as go
import numpy as np


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


def solution_space_similarity(maskA, maskB):
    oc = overlap_coefficient(maskA, maskB)
    J = jaccard_index(maskA, maskB)

    return np.sqrt(J**2 + oc**2) / np.sqrt(2)


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
