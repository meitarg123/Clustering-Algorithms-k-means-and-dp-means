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

class DPMeans:
    def __init__(self, max_iterations=3000, lamda = 1.7):
        self.max_iterations = max_iterations
        self.lamda = lamda

    def fit(self, data_set):
        #initiate the centroids 
        self.centroids_initiation(data_set)
        #optimizing centroids values
        iteration = 1
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration <= self.max_iterations: #len(self.centroids) != len(prev_centroids) or
            points = [[] for _ in range(self.number_of_clusters)]
            for p in data_set:
                distances = euclidean(p, self.centroids)
                centroid_idx = np.argmin(distances)
                distance = np.amin(distances)

                if (distance>self.lamda):
                    self.number_of_clusters += 1
                    self.centroids.append(p)
                    points.append([])
                    points[self.number_of_clusters-1].append(p)
                
                else:
                    points[centroid_idx].append(p)

            # calculate centroids values as the mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in points]
           
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration = iteration + 1
        return self.centroids


    def centroids_initiation(self,data_set):
        self.number_of_clusters=1
        self.centroids=[]
        points =[]
        self.centroid_indexs = []
        for p in data_set:
            self.centroid_indexs.append(0)
            points.append(p)
        self.centroids = [np.mean(points, axis=0)]

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
    centers = 4
    data_set, labels = make_blobs(n_samples=3000, centers=centers, random_state=9) 
    data_set = StandardScaler().fit_transform(data_set)
    # Fit centroids to dataset
    kmeans = DPMeans()
    kmeans.fit(data_set)
    # View results
    class_centers, classification = kmeans.evaluate(data_set)
    visualize(data_set,classification,kmeans,labels)

if __name__=="__main__":
    main()