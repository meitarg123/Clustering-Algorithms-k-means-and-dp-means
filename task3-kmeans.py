import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from kmeans import KMeans
import time

def task3(k):
    img_arr = img.imread('Mandrill-k-means-reduced128.png')
    (h,w,c) = img_arr.shape
    img2D = img_arr.reshape(h*w,c)
    dpmeans_model = KMeans(k)
    centroids = dpmeans_model.fit(img2D)
    class_centers,cluster_labels = dpmeans_model.evaluate(img2D)
    rgb_cols = [[0,0,0] for center in centroids]
    j = 0
    for center in centroids:
        for i in range (len(center)):
            rgb_cols[j][i] = (center[i]*256).round(0).astype(int)
        j = j + 1 

    numpy_rgb_cols = np.array([])
    numpy_rgb_cols = np.array([entity for entity in rgb_cols], np.int32)
    numpy_cluster_labels = np.array(cluster_labels)
    img_quant = np.reshape(numpy_rgb_cols[numpy_cluster_labels],(h,w,c)) # assigning points to the clusters

    return img_quant

if __name__=="__main__":
    start_time = time.time()
    fig, ax = plt.subplots(1,5, figsize=(16,12))
    img_arr = img.imread('Mandrill-k-means.png')


    i=1
    for k in [2,4,8,10]:
        ax[i].imshow(task3(k))
        ax[i].set_title(f"with l value {k}")
        i+=1

    ax[0].imshow(img_arr)
    ax[0].set_title(f"Original Image")
    fig.suptitle(f"execution time: {time.time() - start_time}", fontsize=15)
    plt.show()
