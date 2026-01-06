# Principal Component Analysis

from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas
import plotly.express as px

# Load the iris dataset
iris = load_iris(as_frame=True)
irisDataType = iris['data']
irisData = iris['data']
irisTarget = iris['target']
irisDataType['type'] = irisTarget

print(irisDataType)
print('--------')
# print(irisTarget.describe())
# irisTarget.describe()

# Perform PCA, treinando o alg
pca = PCA(n_components=3)
pca.fit(irisData)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

X_pca = pca.transform(irisData)

result_pca = pandas.DataFrame(X_pca)
result_pca.columns = ['PCA1', 'PCA2', 'PCA3']
result_pca['type'] = irisTarget

chart_3d = px.scatter_3d(result_pca, x='PCA1', y='PCA2', z='PCA3', color='type', title='PCA with k = 3')
chart_3d.show()