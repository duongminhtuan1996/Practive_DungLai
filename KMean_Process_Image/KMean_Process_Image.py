import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

img = plt.imread('Image/a.jpg')

width = img.shape[0]
height = img.shape[1]
deep = img.shape[2]

img = img.reshape(width*height, deep)

#Train and predict algorithm Kmeans
kmeans = KMeans(n_clusters=4).fit(img)
labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

#Method 1
img2 = np.zeros_like(img)
for i in range(len(img2)):
    img2[i] = clusters[labels[i]]

img2 = img2.reshape(width, height, deep)

#Method 2
img3 = np.zeros((width, height, deep), dtype=np.uint8)
index = 0
for i in range(width):
    for j in range(height):
        label_of_pixel = labels[index]
        img3[i][j] = clusters[label_of_pixel]
        index += 1

plt.imshow(img3)
plt.show()