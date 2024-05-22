# Clustering Algorithms: k-means and dp-means

This project involves implementing and analyzing the k-means and dp-means clustering algorithms, particularly focusing on their performance on image data when varying the k and λ values.

## k-means Algorithm

### Section 1: Description of the Algorithm and Its Implementation

The k-means algorithm is a widely used clustering technique that aims to partition a dataset into k clusters, where each data point belongs to the cluster with the nearest mean value.

#### Implementation Steps:
1. **Initialization**: 
   - Randomly initialize k points (called "centroids") within the range of the minimum and maximum x and y values of the dataset.
   - The dataset is generated using the `make_blobs` method from the `sklearn` library.

2. **Iteration**:
   - **Distance Calculation**: Compute the Euclidean distance between each data point and each centroid.
   - **Assignment**: Assign each data point to the nearest centroid, forming k clusters.
   - **Centroid Update**: Recalculate the centroids as the mean of all points assigned to each cluster.
   - **Convergence Check**: Repeat the above steps until the centroids do not change between iterations or a maximum number of iterations is reached to prevent infinite loops.

The full implementation can be found in the attached Python file.

### Section 2: Examining the Implementation Performance

While the k-means algorithm generally performs well, we observed instances where the algorithm's performance was suboptimal. This was primarily due to the random initialization of centroids. We identified two main issues affecting performance:

In the following graphs, we colored the points we created using make_blobs in a different color, and each point is represented by a **different shape** according to the k-means classification

### Initial Central Blob Issue
If a central blob is initialized far from the groups created by `make_blobs`, it tends not to change its position, leading to errors. For example, in the figure below, the lower right plus sign represents a central point at the end of the run with coordinates \( x = 1.292 \), \( y = -1.725 \). Its initial coordinates were almost the same: \( x = 1.289 \), \( y = -1.724 \).

![Initial Central Blob](https://github.com/meitarg123/K-means/assets/100788290/91b6196e-c443-4b12-b6a6-0f698157500f)

### Close Initialization of Central Points

If two center points are initialized too close to each other, they probably won't move away from each other. The red plus signs are the initial values of the central points, while the black plus signs are the final values. As shown below, the points on the lower left side of the image did not move away, causing one group (green) to be divided into two different clusters.

![Close Initialization](https://github.com/meitarg123/K-means/assets/100788290/bee14e82-131d-4e62-8828-c0371b0538d7)


From testing the effect of the value of parameter \( k \) on the performance of the algorithm, we noticed that the performance of k-means deteriorates as \( k \) increases. This is likely due to a higher chance of problematic initialization of the central points, leading to the issues described above. Below is an example of the grouping of 3 different central points that were initialized with close values.

We emphasize that the k-means algorithm is suitable for classification problems where the number of different groups is known in advance. Therefore, when creating the dataset using `sklearn.datasets.make_blobs`, we create a number of groups corresponding to the parameter \( k \). As required, we scattered 1000 points in a two-dimensional space.

![Effect of k](https://github.com/meitarg123/K-means/assets/100788290/c0642350-6817-44ca-9e52-0b1fccb6e7c8)

## Improved Initialization Method

To address these problems, we decided to initialize the \( k \) points differently:

1. Initialize the values of the first central point to be the same as one of the existing points in the dataset, chosen at random.
2. For the remaining \( k-1 \) central points, use the following method:
    1. For each point in the dataset, calculate the sum of its distances to all the central points calculated so far. Normalize this distance against the sum of all distances of all points from all centers.
    2. Initialize the values of the next central point by drawing a random point from the dataset, with the probability of drawing a particular point proportional to the sum of its normalized distances.

With this approach, the performance improves because the initial central point values necessarily match some points from the dataset, and the probability of initializing two central points too close to each other is reduced.

# dp-means algorithm

## Description

Unlike k-means, the dp-means algorithm doesn't require the number of clusters to be known in advance, making it suitable for various types of problems. The algorithm determines the number of clusters during runtime using a parameter λ.

### Algorithm Steps:

1. Initialize a single central point by averaging all points in the dataset.
2. Iterate through the dataset:
   - Calculate the distance of each point from each central point.
   - Find the minimum distance for each point.
   - If the minimum distance is greater than λ, create a new central point.
3. Repeat the process until convergence, where no new central points are created or the number of central points remains constant between iterations.

## Performance Considerations

The value of λ represents the maximum Euclidean distance of a point from its nearest central point.

- A small λ value (< half the distance between any two points) results in each point becoming a central point.
- A large λ value may classify all points into one central point.

The optimal λ depends on:

1. The number of groups in the dataset:
   - Larger λ values work better with fewer groups.
2. The distribution of points in space:
   - Larger λ values optimize central points towards the center of large clusters.
   - Smaller λ values create additional central points for large clusters.

## Examples

### Example 1: λ = 1 (2 groups)

![Example 1](https://github.com/meitarg123/K-means/assets/100788290/c382d8d3-1b78-4431-88c9-be5ca4c1852a)

### Example 2: λ = 2 (2 groups)

![Example 2](https://github.com/meitarg123/K-means/assets/100788290/6362fba9-c0af-4e48-90de-427120c150da)

### Example 3: λ = 1.2 (4 groups)

![Example 3](https://github.com/meitarg123/K-means/assets/100788290/a4afa174-d141-441d-b569-286994ae8eb1)

### Example 4: λ = 1.7 (4 groups)

![Example 4](https://github.com/meitarg123/K-means/assets/100788290/df3c0b60-2676-46a0-8e62-95873e56a4d1)

# Section 3: Image Clustering- Comparing K-means and DP-means

## K-means

### Description

In the K-means algorithm applied to the Madril monkey image (512\*512), increasing the value of K resulted in better performance. However, for high K values (30 and up), there was no significant change in performance.

### Observations

- Grouping of central points in the same group of close points in the data-set and their (unnecessary) division into different clusters.
- Locating a central point far from any point in the data-set.

These mistakes did not significantly affect the performance of the algorithm in the Mandrill monkey coloring task because proximity between two central points resulted in the same (or sufficiently similar) coloring of the output image. Also, a central point located far from any point in the data-set did not result in significant miscoloring (if any) of pixels in the output image.

### Running Time

Higher K values predictably resulted in longer running times. For the first 4 images, the runtime was 9.4 seconds. For the last 4 images, the running time was 27.35 seconds.

## DP-means

### Description

In the DP-means algorithm, lowering the value of λ resulted in better performance. Lowering λ allows the algorithm to create more central points and classify more points in the dataset into more clusters as needed.

### Observations

- A high λ value resulted in the creation of few central points, leading to the classification of the points in the dataset into a small number of clusters.
- Lowering the value of λ resulted in more iterations and central points creation, increasing running time but without significant impact on performance.

### Examples

- High λ value: Few central points, resulting in less diverse output image.
- Low λ value: More iterations, more central points, closer to original image.

![High λ Value](https://github.com/meitarg123/K-means/assets/100788290/2c68e525-f64e-43a5-a295-cd5e6f56c934)
![Low λ Value](https://github.com/meitarg123/K-means/assets/100788290/551302ae-1d40-4d61-a38b-cfd9f5202897)

