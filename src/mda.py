from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import circstd


def distance(
    point1: NDArray,
    point2: NDArray,
    dir_indices: Optional[list[int]],
    std_list: list[float],
) -> float:
    vector = np.zeros_like(point1)
    for ind, (p1, p2) in enumerate(zip(point1, point2)):
        div = np.sqrt(2) * std_list[ind]
        if dir_indices is not None and ind in dir_indices:
            diff = ((p1 - p2) + 180) % 360 - 180
            vector[ind] = diff / div
        else:
            vector[ind] = abs(p1 - p2) / div
    return float(np.linalg.norm(vector))


def distance2subset(data_point, subset, dir_indices, std_list) -> float:
    min_dist = np.inf
    for point in subset:
        dist = distance(data_point, point, dir_indices, std_list)
        if dist < min_dist:
            min_dist = dist
    return min_dist


def get_std(
    data_tuple: tuple[NDArray, ...],
    dir_indices: Optional[list[int]],
) -> list[float]:
    std_list = []
    for ind, array in enumerate(data_tuple):
        if dir_indices is not None and ind in dir_indices:
            std = circstd(array, high=360, low=0)
        else:
            std = np.std(array)
        std_list.append(std)
    return std_list


def max_diss_alg(
    data_tuple: tuple[NDArray, ...],
    seed_index: int = 0,
    n_clusters: int = 10,
    dir_indices: Optional[list[int]] = None,
) -> NDArray:
    """Maximum dissimilarity clustering algorithm"""

    # Preprocess the data tuple
    data: NDArray = np.array(data_tuple, dtype=np.float32).T
    std_list = get_std(data_tuple, dir_indices)
    n_inputs = len(data_tuple)

    # Initialize the centroids with the seed
    subset: NDArray = np.array([data[seed_index]])

    while len(subset) < n_clusters:
        max_dist = -1
        next_point = None
        # Find the point with the maximum distance to the subset
        for i, data_point in enumerate(data):
            if data_point in subset:
                continue
            # Get the minimum distance to a point in the subset
            min_dist = distance2subset(data_point, subset, dir_indices, std_list)
            if min_dist > max_dist:
                max_dist = min_dist
                next_point = data_point
            # Compute the progress of the for loop percentage
            progress = (i + 1) / len(data) * 100
            print(f"Finding cluster nº{len(subset)} --> {progress:.2f}%", end="\r")
        # Add the point with the maximum distance to the subset
        if next_point is not None:
            subset = np.append(subset, next_point).reshape(-1, n_inputs)

    return subset
