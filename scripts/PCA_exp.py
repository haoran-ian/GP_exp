import numpy as np
from sklearn.decomposition import PCA


def PCA_train(dim, lb, ub):
    X_pca_train = np.random.uniform(lb, ub, size=(10000*dim, lb.shape[0]))
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(X_pca_train)
    print(f"{pca.n_components_}")


dim = 45
lb = np.array([-1. for _ in range(dim)])
ub = np.array([1. for _ in range(dim)])
PCA_train(dim, lb, ub)
