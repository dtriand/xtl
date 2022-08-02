import numpy as np


class KMeans:

    def __init__(self, n_clusters: int = 8, initializer: str = 'random', max_iter: int = 300, tol: float = 1e-4,
                 seed = None, debug = False):
        self.n_clusters = n_clusters
        self.initializer = initializer
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.debug = debug
        self.history = {
            'centroids': []
        }

        self.cluster_centers: np.ndarray
        self.data: np.ndarray
        # self.inertia: float
        self.n_iter: int
        # self.n_features: int

    def _initialize_centroids(self, X: np.ndarray):
        if self.initializer == 'random_samples':
            np.random.RandomState(self.seed)
            ri = np.random.permutation(X.shape[0])
            self.centroids = X[ri[:self.n_clusters]]
            return self.centroids
        elif self.initializer == 'random':
            np.random.RandomState(self.seed)
            cx = np.random.uniform(X.T[0].min(), X.T[0].max(), (self.n_clusters,))
            cy = np.random.uniform(X.T[1].min(), X.T[1].max(), (self.n_clusters,))
            self.centroids = np.vstack([cx, cy]).T
            return self.centroids
        else:
            raise NotImplemented(self.initializer)

    def _compute_centroids(self, X: np.ndarray, labels):
        self.centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            self.centroids[k, :] = np.mean(X[labels == k, :], axis=0)
        return self.centroids

    def _compute_distance(self, X: np.ndarray, centroids: np.ndarray):
        self.distance_array = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            self.distance_array[:, k] = np.square(np.linalg.norm(X - centroids[k, :], axis=1))
        return self.distance_array

    def _find_closest_cluster(self, distance_array):
        return np.argmin(distance_array, axis=1)

    def _compute_sse(self, X, labels, centroids):
        distance = np.zeros(X.shape[0])
        for k in range(self.n_clusters):
            distance[labels == k] = np.linalg.norm(X[labels == k] - centroids[k], axis=1)
        return np.sum(np.square(distance))

    def fit(self, X: np.ndarray):
        self.data = X
        self._initialize_centroids(self.data)
        for self.n_iter in range(self.max_iter):
            old_centroids = self.centroids
            if self.debug:
                self.history['centroids'].append(old_centroids)
            distance_array = self._compute_distance(self.data, old_centroids)
            self.labels = self._find_closest_cluster(distance_array)
            self.centroids = self._compute_centroids(self.data, self.labels)
            if np.all(np.abs(old_centroids - self.centroids) <= self.tol):
                break

    def fit2(self, X: np.ndarray):
        self.data = X
        self._initialize_centroids(X)
        for self.n_iter in range(self.max_iter):
            print(f'iter {self.n_iter}')
            old_centroids = self.centroids
            distance = self._compute_distance(X, old_centroids)
            self.labels = self._find_closest_cluster(distance)
            self.centroids = self._compute_centroids(X, self.labels)
            if np.all(np.abs(old_centroids - self.centroids) <= self.tol):
                break

    def transform(self, X):
        ...

    def fit_transform(self, X):
        ...

    def plot(self):
        ...

    def plot_evolution(self):
        ...