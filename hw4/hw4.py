import csv
import geopandas
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt


def load_data(filepath):
    with open(filepath, "r", newline = '', encoding = "utf-8") as file:
        csv_data = csv.DictReader(file, delimiter=',')
        data = [dict(row) for row in csv_data]
    return data

def calc_features(row):
    return_array = np.array([])
    row_iter = iter(row.items())
    next(row_iter)
    for key, value in row_iter:
        return_array = np.append(return_array, np.float64(value))
    return return_array


def hac(features):
    n = len(features)
    features = [np.array(f) for f in features]  # Ensure all features are numpy arrays

    Z = np.zeros((n - 1, 4))

    clusters = {i: [i] for i in range(n)}
    cluster_sizes = {i: 1 for i in range(n)}
    active_clusters = set(range(n))

    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            distances[(i, j)] = np.linalg.norm(features[i] - features[j])
            distances[(j, i)] = distances[(i, j)]

    # Perform n-1 merges
    for i in range(n - 1):
        min_dist = np.inf
        min_pair = None

        for ci in active_clusters:
            for cj in active_clusters:
                if ci < cj:
                    max_pairwise_dist = -np.inf
                    for pi in clusters[ci]:
                        for pj in clusters[cj]:
                            if pi < n and pj < n:
                                curr_dist = distances[(pi, pj)]
                                max_pairwise_dist = max(max_pairwise_dist, curr_dist)

                    if max_pairwise_dist < min_dist:
                        min_dist = max_pairwise_dist
                        min_pair = (ci, cj)
                    elif max_pairwise_dist == min_dist:
                        if ci < min_pair[0]:
                            min_pair = (ci, cj)
                        elif ci == min_pair[0] and cj < min_pair[1]:
                            min_pair = (ci, cj)

        cluster1, cluster2 = min_pair

        new_cluster_idx = n + i
        clusters[new_cluster_idx] = clusters[cluster1] + clusters[cluster2]
        cluster_sizes[new_cluster_idx] = cluster_sizes[cluster1] + cluster_sizes[cluster2]

        active_clusters.remove(cluster1)
        active_clusters.remove(cluster2)
        active_clusters.add(new_cluster_idx)

        Z[i, 0] = cluster1
        Z[i, 1] = cluster2
        Z[i, 2] = min_dist
        Z[i, 3] = cluster_sizes[new_cluster_idx]

    return Z

def fig_hac(Z, names):
    fig = plt.figure()
    sch.dendrogram(Z, labels = names, leaf_rotation = 90)
    plt.tight_layout()
    plt.show()
    return fig

def normalize_features(features):
    data = np.array(features)

    means = np.mean(data, axis = 0)
    stds = np.std(data, axis = 0)

    normalized = (data - means) / stds

    normalized_vectors = [np.array(vec) for vec in normalized]

    return normalized_vectors


def world_map(Z, names, K_clusters):
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    world['name'] = world['name'].str.strip()
    names = [name.strip() for name in names]

    world['cluster'] = np.nan

    n = len(names)
    clusters = {j: [j] for j in range(n)}

    for step in range(n-K_clusters):
        cluster1 = Z[step][0]
        cluster2 = Z[step][1]

        # Create new cluster id as n + step
        new_cluster_id = n + step

        # Merge clusters
        clusters[new_cluster_id] = clusters.pop(cluster1) + clusters.pop(cluster2)

    # Assign cluster labels to countries in the world dataset
    for i, value in enumerate(clusters.values()):
        for val in value:
            world.loc[world['name'] == names[val], 'cluster'] = i

    # Plot the map
    world.plot(column='cluster', legend=True, figsize=(15, 10), missing_kwds={
        "color": "lightgrey",  # Set the color of countries without clusters
        "label": "Other countries"
    })

    # Show the plot
    plt.show()