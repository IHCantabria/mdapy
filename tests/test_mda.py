import numpy as np

from mdapy.src.mda import distance

# Common variable used in tests
std_list = [1 / np.sqrt(2)] * 2


def test_distance_for_scalar():
    point1 = np.array([0, 0])
    point2 = np.array([1, 1])
    result = distance(point1, point2, None, std_list)
    assert result == np.sqrt(2)


def test_distance_for_directional():
    point1 = np.array([0, -5])
    point2 = np.array([10, 5])
    result = distance(point1, point2, [0, 1], std_list)
    assert result == np.sqrt(200)


def test_distance_for_scalar_and_directional():
    point1 = np.array([1, -5])
    point2 = np.array([-1, 5])
    result = distance(point1, point2, [0, 1], std_list)
    assert result == np.sqrt(104)
