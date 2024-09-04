################################################
# Created By  : Pablo Garcia (for IHCantabria)
# Created Date: 18/09/2023
# Version     : 0.1
################################################

from functools import cache
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.stats import circstd


def distance(point1: NDArray, point2: NDArray, dir_indices: Optional[list[int]], std_list: list[float]) -> float:
    vector = np.zeros_like(point1)
    for ind, (p1_ind, p2_ind) in enumerate(zip(point1, point2)):
        if dir_indices is not None and ind in dir_indices:
            dist = ((p1_ind - p2_ind) + 180) % 360 - 180
        else:
            dist = p1_ind - p2_ind
        vector[ind] = dist / (np.sqrt(2) * std_list[ind])
    return float(np.linalg.norm(vector))


def preprocess(data_tuple: tuple[NDArray, ...]) -> NDArray:
    return np.array(data_tuple, dtype=np.float32).T


def std_list(data_tuple: tuple[NDArray, ...], dir_indices: Optional[list[int]]) -> list[float]:
    return_list = []
    for ind, array in enumerate(data_tuple):
        if dir_indices is not None and ind in dir_indices:
            computed_std = circstd(array, high=360, low=0)
        else:
            computed_std = np.std(array)
        return_list.append(computed_std)
    return return_list


class DataMatrix:
    def __init__(self, data_tuple: tuple[NDArray, ...], dir_indices: Optional[list[int]]):
        # Check if any of the arrays in the tuple has NaN values
        for array in data_tuple:
            if np.isnan(array).any():
                raise ValueError("The data contains NaN values")
        self.data = preprocess(data_tuple)
        self.std_list = std_list(data_tuple, dir_indices)
        self.dir_indices = dir_indices

    def __getitem__(self, index: list[int] | int) -> NDArray:
        return self.data[index]

    @property
    def num_points(self) -> int:
        return len(self.data)

    @cache
    def distance_between(self, index1: int, index2: int) -> float:
        point1, point2 = self.data[[index1, index2]]
        return distance(point1, point2, self.dir_indices, self.std_list)


def distance_to_subset(data_matrix: DataMatrix, point_index: int, subset_indices: list[int]) -> float:
    distance_list = [data_matrix.distance_between(point_index, i_point) for i_point in subset_indices]
    return min(distance_list)


def max_diss_alg(
    data_tuple: tuple[NDArray, ...],
    n_clusters: int,
    seed_index: int = 0,
    dir_indices: Optional[list[int]] = None,
) -> tuple[NDArray, list[int]]:
    """
    This function implements the Maximum Dissimilarity Algorithm (MDA) to cluster a data.

    Parameters
    ----------
    data_tuple : tuple[NDArray, ...]
        A tuple of arrays with the data to cluster.
    n_clusters : int
        The number of clusters to generate.
    seed_index : int, optional
        The index of the seed point, by default 0
    dir_indices : list[int], optional
        The indices of directional arrays in the tuple, by default None

    Returns
    -------
    tuple[NDArray, list[int]]
        A tuple with the clustered data points and their indices in the original data.
    """

    # Create the data matrix
    data_matrix = DataMatrix(data_tuple, dir_indices)

    # Initialize the centroids with the seed
    subset_indices: list[int] = [seed_index]

    while len(subset_indices) < n_clusters:
        max_dist = next_point_index = -1
        # Find the point with the maximum distance to the subset
        for i_point in range(data_matrix.num_points):
            if i_point in subset_indices:
                continue
            # Get the distance to the closest point in the subset
            distance = distance_to_subset(data_matrix, i_point, subset_indices)
            if distance > max_dist:
                max_dist = distance
                next_point_index = i_point
            # Compute the progress of the for loop percentage
            progress = (i_point + 1) / data_matrix.num_points * 100
            print(f"Finding cluster nº{len(subset_indices)+1} --> {progress:.2f}%", end="\r")
        # Add the point with the maximum distance to the subset
        if next_point_index > -1:
            subset_indices.append(next_point_index)

    return data_matrix[subset_indices], subset_indices
