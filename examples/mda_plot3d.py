import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from mdapy import max_diss_alg

# Load wave data (140_256 hours)
datawave = np.load("examples/data/datawave.npz")

# Generate a random sample of N_SHORT data points
indices = np.random.randint(0, 140_256, N_SHORT := 10_000)
dir: NDArray = datawave["dir"][indices]
tp: NDArray = datawave["tp"][indices]
hs: NDArray = datawave["hs"][indices]

# Select the seed point index and the number of clusters
seed_index = np.argmax(tp).astype(int)
n_clusters = 10

# Cluster the data array (2D)
clustered_data = max_diss_alg((hs, tp, dir), n_clusters, seed_index, dir_indices=[2])

# Plot the data points and the clustered ones
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(tp, dir, hs, c="b", marker=".", alpha=0.1)
ax.scatter(clustered_data[:, 1], clustered_data[:, 2], clustered_data[:, 0], c="r", marker="x")
plt.show()
