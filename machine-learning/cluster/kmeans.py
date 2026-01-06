# K-means is a popular, unsupervised machine learning algorithm used for partitioning
# unlabeled data into a pre-defined number (k) of distinct clusters, grouping similar
# data points together based on proximity to cluster centers (centroids).

# You MUST select the number of the clusters (K)

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas
import plotly.express as px


iris = load_iris(as_frame=True)
irisDataType = iris['data']
irisData = iris['data']
irisTarget = iris['target']
irisDataType['type'] = irisTarget


irisDataType.describe()

kmeans = KMeans(n_clusters=3)

kmeans.fit(irisData)

# Result
print(kmeans.labels_)
# print(kmeans.cluster_centers_)

irisDataType['cluster'] = kmeans.labels_
irisDataType.groupby(['type','cluster']).count()

fig = px.scatter(irisDataType, x='sepal length (cm)', y='sepal width (cm)', color='cluster', title='KMeans - k = 3')
fig.show()

