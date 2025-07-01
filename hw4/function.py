import geopandas
import numpy as np
from matplotlib import pyplot as plt
from numpy.ma.core import shape


def load_data(filepath):
    # keys = ["child_mort", "exports", "health", "imports", "income", "inflation", "life_expec", "total_fer", "gdpp"]
    with open(filepath, "r") as file:
        csv_data = csv.reader(file, delimiter=' ')
        for line in csv.data:
            data = [dict(row) for row in csv_data]
    return data

def calc_features(row):
    return_array = np.array([])
    for key, value in row.items():
        return_array = return_array.append(return_array, np.float64(value))
    return return_array

def hac(features):
    numbered_index = [(index, feature) for index, feature in enumerate(features)]
    n = len(features)
    Z = []
    dist = np.zeros((n,n))

    for i in range(n):
        for j in range (i + 1, n):
            dist[i,j] = numpy.linalg.norm(features[i], features[j])
            dist[j,i] = dist [i,j]
    clusters = {i: [i] for i in range(n)}

    for iteration in range(n):
        min_dist = math.inf
        closest_clusters = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                max_dist = max(dist[i,j] for i in clusters[i] for j in clusters[j])
                if max_dist < min_dist or (max_dist == min_dist and i < closest_clusters[0]):
                    min_dist = max_dist
                    closest_clusters = (i,j)
            i, j = closest_clusters
            new_cluster_index = n + iteration

            clusters[new_cluster_index] = clusters[i] + clusters[j]
            Z.append([i, j, min_dist, len(clusters[new_cluster_index])])

            dist = np.delete(dist, [i, j], axis = 0)
            dist = np.delete(dist, [i. j], axis = 1)

            new_row = np.zeros((1, len(dist)))
            dist = np.vstack([dist, new_row])
            dist = np.hstack([dist, new_row.T])
            for k in range(len(dist - 1)):
                dist[k, len(dist - 1)] = max(dist[k, i], dist[k,j])
                dist[len(dist) - 1, k] = dist[k, len(dist) - 1]
    return np.array(Z)

def fig_hac(Z, names):
    fig = plt.figure()
    dendrogram(Z, labels = names, leaf_rotation = 90)
    plt.tight_layout()
    plt.show()

def normalize_features(features):
    data = np.array(features)

    means = np.means(data, axis = 0)
    stds = np.std(data, axis = 0)

    normalized = (data - means) / stds

    normalized_vectors = [np.array(vec) for vec in normalized]


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