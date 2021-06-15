#Navreet Kaur 
#Machine learning Project
#original file from the Code Scholar



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Take make moons in built dataset

data_ = datasets.make_moons(200)

# check the dataset

data_

# create input dataframe
inputData = pd.DataFrame(data = data_[0])

inputData.head()

# create output dataframe

outputData = pd.DataFrame(data = data_[1])
outputData.head()

# create a scatter plot for inputData set
plt.scatter(inputData[0], inputData[1])

# create a scatter plot for inputData set with outputData color

outputData = pd.DataFrame(data = data_[1])
outputData.head()
plt.scatter(inputData[0], inputData[1], c = outputData)



# Call the sklearn Kmeans and make a model with 200 samples
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5)
model.fit(inputData)

#model_fit

# check for labels

model.labels_



# call metrics and check silhoutte score
from sklearn import metrics
metrics.silhouette_score(inputData, model.labels_)

# create a scatter plot for inputData set with model labels color

plt.scatter(inputData[0], inputData[1], c = model.labels_)

"""#### finding right number of cluster"""

cluster_range = range(1, 20)
error_list = []

for i in cluster_range:
    model = KMeans(n_clusters=i)
    model.fit(inputData)
    res = model.inertia_
    error_list.append(res)

import matplotlib.pyplot as plt

plt.plot(cluster_range, error_list, marker = "o", color = "g", markersize = 10)
plt.xlabel("Cluster Range")
plt.ylabel("IntraCluster Sum")
plt.title("KMeans")
plt.show()



