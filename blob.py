import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=100, centers=3, cluster_std=0.4, random_state=0)
#plt.scatter(X[:,0], X[:,1])
model = KMeans(n_clusters=3)
model.fit(X)
prediction=model.predict(X)
S=model.score(X)

X_test=np.array([[2,4.5]])
Y_test=model.predict(X_test)
# plt.scatter(X[:,0], X[:,1], c=model.predict(X))
# plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='r')
# plt.scatter(X_test[0,0],X_test[0,1])


from sklearn.ensemble import IsolationForest
X, y = make_blobs(n_samples=50, centers=1, cluster_std=0.1, random_state=0)
X[-1,:] = np.array([2.25, 5])

#plt.scatter(X[:,0], X[:, 1])
model = IsolationForest(contamination=0.01)
model.fit(X)

#plt.scatter(X[:,0], X[:, 1], c=model.predict(X))

from sklearn.datasets import load_digits

digits = load_digits()
images = digits.images
X = digits.data
y = digits.target

#plt.imshow(images[42])
model = IsolationForest(random_state=0, contamination=0.02)
model.fit(X)
prediction2=model.predict(X)
outliers = model.predict(X) == -1 

# plt.figure(figsize=(12, 3))
# for i in range(10):
#   plt.subplot(1, 10, i+1)
#   plt.imshow(images[outliers][i])
#   plt.title(y[outliers][i])
  
  
  
from sklearn.decomposition import PCA

model = PCA(n_components=2)
model.fit(X)
x_pca = model.transform(X)
# plt.scatter(x_pca[:,0], x_pca[:,1], c=y)
# plt.colorbar()
#ou encore
# plt.figure()
# plt.xlim(-30, 30)
# plt.ylim(-30, 30)

# for i in range(100):
#     plt.text(x_pca[i,0], x_pca[i,1], str(y[i]))
    

n_dims = X.shape[1]
model = PCA(n_components=n_dims)
model.fit(X)

variances = model.explained_variance_ratio_

meilleur_dims = np.argmax(np.cumsum(variances) > 0.90)


plt.bar(range(n_dims), np.cumsum(variances))
plt.hlines(0.90, 0, meilleur_dims, colors='r')
plt.vlines(meilleur_dims, 0, 0.90, colors='r')





















