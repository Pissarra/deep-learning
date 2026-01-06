# DBSCAN - Density-Based Spatial Clustering of Applications with Noise

# a clustering algorithm that groups together closely packed data points
# into dense regions, treating points in sparse areas as outliers (noise)

# You DONT select the number of the clusters (K)
# The values that are in no cluster, will be categorized as noise

from sklearn.datasets import load_iris
from sklearn.cluster import DBSCAN
import pandas
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris(as_frame=True)
irisDataType = iris['data']
irisData = iris['data']
irisTarget = iris['target']
irisDataType['type'] = irisTarget

irisDataType.describe()

dbscan = DBSCAN(eps=1, min_samples=5)
dbscan.fit(irisData)
# dbscan = DBSCAN(eps=0.5, min_samples=1).fit(irisData)

# Result
print(dbscan.labels_)

irisDataType['cluster'] = dbscan.labels_
irisDataType.groupby(['type','cluster']).count()

# fig = px.scatter(irisDataType, x='sepal length (cm)', y='sepal width (cm)', color='cluster', title='DBSCAN - EPS = 1 - Min Samples = 5')
# fig.show()

# Deixar os dados na mesma escala, aplicar o PCA e testar o DBSCAN
scaler = StandardScaler()
x_scaled = scaler.fit_transform(irisData)

pca_scaled = PCA(n_components=2)
pca_scaled.fit(x_scaled)

print('PCA scaled')
print(pca_scaled.explained_variance_ratio_)
print('Percentage of variance explained: ' + str(sum(pca_scaled.explained_variance_ratio_)))

# Result 2
result_pca_scaled = pandas.DataFrame(pca_scaled.transform(x_scaled))
result_pca_scaled.columns = ['PCA1', 'PCA2']

dbscan_pca = DBSCAN(eps=1, min_samples=5)
dbscan_pca.fit(x_scaled)

result_pca_scaled['cluster'] = dbscan_pca.labels_

fig = px.scatter(result_pca_scaled, x='PCA1', y='PCA2', color='cluster', title='DBSCAN - EPS = 1 - Min Samples = 5')
fig.show()
