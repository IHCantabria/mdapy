# 👋 Welcome to the mdapy repository

## ❗ Instalation Note
This package name on Github is `mdapy`, but the package name on Pypi is `ihc-mdapy` due to a name conflict. So, if you want to install the package, you should use:
```bash
pip install ihc-mdapy
```

---

## 📋 The Maximum Dissimilarity Algorithm
The Maximum Dissimilarity Algorithm is a technique used to select a specified number of data points from a larger dataset in such a way that the selected points are maximally dissimilar to each other. The algorithm is particularly useful in scenarios where you need to create diverse subsets of data for analysis or for initializing cluster centroids in clustering algorithms.

In this case, directional variables can also be considered, in which case, the distance between two angles is calculated as the shortest distance between the two angles in a circle. For example, the distance between 350° and 10° is 20°, not 340°. This is the approach used in this implementation of the algorithm (see [Guanche et al. (2013)](#1) for more details).

### Algorithm Steps
1. **Initialization**: Start with an empty subset, S. Choose the first data point arbitrarily or based on some criteria and add it to S.

2. **Iteration**: Repeat the following steps until you have selected the desired number of data points (n).

3. **Finding the Most Dissimilar Point**: For each data point not in S, calculate the minimum distance from that point to the points in S.

4. **Selecting the Maximum Dissimilarity Point**: Choose the data point with the greatest dissimilarity (maximum distance calculated in the previous step) and include it in subset S.

5. **Output**: Return the subset S, which contains n data points that are maximally dissimilar to each other.

---

## 💻 Installation
You can use the Maximum Dissimilarity Algorithm by following these steps:

1. Install the mdapy package using pip:
   ```bash
   pip install ihc-mdapy
   ```

   Adiotionally, you can also install the package from the source code:
   ```bash
   git clone https://github.com/IHCantabria/mdapy
   cd mdapy
   pip install .
   ```

1. Import the max_diss_alg function from the mdapy package:
   ```python
   from mdapy import max_diss_alg
   ```

1. Create a tuple with the data arrays to use (must have same lengths) and choose the number of data points to select:
   ```python
   tp = [14.93, 10.42, 7.46, 7.63, 12.20, 12.05, 14.49, 8.62]
   hs = [1.82, 1.45, 0.26, 0.90, 1.30, 5.71, 1.40, 2.18]
   dr = [268, 272, 289, 293, 276, 273, 276, 292]
   data = (tp, hs, dr)
   n_clusters = 3
   ```

1. Aditionally, choose the seed point to start the algorithm (optional) and, in case any of the variables is directional, provide the position on the tuple of that array.
    ```python
    seed_index = np.argmax(tp)
    dir_indices = [2]
   ```

1. Call the max_diss_alg function and store the output data:
    ```python
    clustered_data, indices = max_diss_alg(data, n_clusters, seed_index, dir_indices)
   ```

---

## ©️ Credits
Developed and maintained by 👨‍💻 [Pablo Garcia](https://github.com/pablofueros) @ 🏢 [IHCantabria](https://github.com/IHCantabria).

---

## 📚 References
<a id="1">[1]</a> Guanche, Y., Guanche, R., Camus, P., Mendez, F. J., & Medina, R. (2013). A multivariate approach to estimate design loads for offshore wind turbines. Wind Energy, 16(7), 1091-1106. (Apen)

---