# Kernel Principal Component Analysis

from sklearn.decomposition import KernelPCA
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
pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.01)
pca.fit(irisData)
# print(pca.explained_variance_ratio_)
# print(sum(pca.explained_variance_ratio_))

X_pca = pca.transform(irisData)

result_pca = pandas.DataFrame(X_pca)
result_pca.columns = ['KPCA1', 'KPCA2']
result_pca['type'] = irisTarget

chart_3d = px.scatter(result_pca, x='KPCA1', y='KPCA2', color='type', title='KPCA with k = 2, Kernel RBF')
chart_3d.show()
chart_3d.show()
