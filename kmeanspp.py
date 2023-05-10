import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random

def euclidean(p, data_set):
    ans = np.sqrt(np.sum((p - data_set)**2, axis=1))
    return ans

class KMeans:
    def __init__(self, number_of_clusters=5, max_iterations=500):
        self.number_of_clusters = number_of_clusters
        self.max_iterations = max_iterations
    def fit(self, data_set):
        #initiate the centroids 
        self.centroids_initiation(data_set)

        #optimizing centroids values
        iteration = 1
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration <= self.max_iterations:
            points = [[] for _ in range(self.number_of_clusters)]
            for p in data_set:
                distances = euclidean(p, self.centroids)
                centroid_idx = np.argmin(distances)
                points[centroid_idx].append(p)
            # calculate centroids values as the mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration = iteration + 1

    def centroids_initiation(self,data_set):
        # initiate centroids using kmeans++ logic
        self.centroids = [random.choice(data_set)]
        for i in range(self.number_of_clusters-1):
            # for each point, calculates its distance to centroids
            distances = np.sum([euclidean(centroid, data_set) for centroid in self.centroids], axis=0)
            # Normalize distances, preparing the values for propability calculation
            distances = distances/np.sum(distances)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(data_set)), size=1, p=distances)
            self.centroids = self.centroids + [data_set[new_centroid_idx]]

    def evaluate(self, data_set):
        centroid_indexs = []
        centroids = []
        for p in data_set:
            distances = euclidean(p, self.centroids)
            centroid_idx = np.argmin(distances)
            centroids.append(self.centroids[centroid_idx])
            centroid_indexs.append(centroid_idx)
        return centroids, centroid_indexs

def visualize(data_set,classification,kmeans,labels):
    sns.scatterplot(x=[X[0] for X in data_set],
        y=[X[1] for X in data_set],
        hue=labels,
        style=classification,
        palette="deep",
        legend=None
        )
    plt.plot([x for x, _ in kmeans.centroids],
        [y for _, y in kmeans.centroids],
        'k+',
        markersize=10,
        )
    plt.show()

def main():
    centers = 5
    data_set, labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    data_set = StandardScaler().fit_transform(data_set)
    # Fit centroids to dataset
    kmeans = KMeans(number_of_clusters=centers)
    kmeans.fit(data_set)
    # View results
    class_centers, classification = kmeans.evaluate(data_set)
    visualize(data_set,classification,kmeans,labels)

if __name__=="__main__":
    main()