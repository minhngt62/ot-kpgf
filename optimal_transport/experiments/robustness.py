from ._experiment import Experiment
from ..models._ot import _OT
from ..classifiers import KNN

from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import scipy
import time


class Robustness(Experiment):
    def __init__(
        self,
        model: Dict[int, _OT],
        exp_name: str,
        log_dir: str,
    ):
        super().__init__(model, exp_name, log_dir)
    
    def run(
        self, xs: np.ndarray, xt: np.ndarray, ys: np.ndarray, yt: np.ndarray, 
        model: _OT, **kwargs
    ) -> float:
        def accuracy(preds, labels):
            return np.sum(preds == labels) / labels.shape[0]
        
        a = np.ones(xs.shape[0]) / xs.shape[0]
        b = np.ones(xt.shape[0]) / xt.shape[0]
        knn = KNN(1)
        knn.fit(xt, yt)
        model.fit(xs, xt, a, b, **kwargs)
        xs_t = model.transport(xs, xt)
        ys_pred = knn.predict(xs_t)
        return accuracy(ys_pred, ys)

    def plot(self, **kwargs):
        pass
    
    @classmethod
    def gaussMixture_meanRandom_covWishart(
        cls,
        n: int,
        d: int,
        k: int, # number of clusters
        d_proj: Optional[int] = None,
        means: Optional[np.ndarray] = None,
        covs: Optional[Union[np.ndarray, List]] = None,
        org_noise_level: float = 0,
        dim_noise_level: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        assert n % k == 0, "Each clusters should have equal number of samples."

        # generate random means for each cluster
        if means is None:
            means = np.random.randn(k, d)

        # sample covariance matrices from a Wishart distribution
        if covs is None:
            covs = []
            for _ in range(k):
                cov = scipy.stats.wishart.rvs(df=d, scale=np.eye(d))
                covs.append(cov)

        # generate data points for each cluster
        X, y, K = [], [], []
        gauss_noise = np.random.randn(k, d)
        for i in range(k):
            Xi = np.random.multivariate_normal(means[i], covs[i], size=(n // k - 1))
            Xi = np.concatenate((means[i][None, :], Xi)) + org_noise_level * np.repeat(gauss_noise[i].reshape(1, -1), n//k, axis=0)
            X.append(Xi) # (n, d)
            y.append(i * np.ones(n // k))
            K.append(i * (n // k))
        X, y = np.concatenate(X), np.concatenate(y)

        # randomly project data points to a 5-dimensional subspace
        if d_proj is not None:
            assert d_proj >= d, "The original data should be embedded in a noisy higher-dimensional space."
            if d_proj != d:
                X = Experiment.add_noise_dims(X, d_proj - d, dim_noise_level=dim_noise_level)
        return X, y, K # <-- K ~ centroid indices
    
    @classmethod
    def add_noise_dims(cls, X, n_dims, dim_noise_level=0.1):
        noise = np.random.normal(scale=dim_noise_level, size=(X.shape[0], n_dims))
        X = np.concatenate([X, noise], axis=1) # n x (d + d_noise)
        return X


class Dimensionality(Robustness):
    def __init__(
        self,
        model: Dict[int, _OT],
        exp_name: str,
        log_dir: str,
    ):
        super().__init__(model, exp_name, log_dir)

    def __call__(
        self,
        hyperplane_dim: int = 5,
        max_projected_dim: int = 100,
        freq_projected_dim: int = 5,
        n_components: int = 3,
        cluster_samples_per_dim: int = 5,
        n_keypoints: Optional[int] = 3,
        source_means: Optional[np.ndarray] = None,
        target_means: Optional[np.ndarray] = None,
        dim_noise_level: float = 1,
    ) -> Dict:
        self.record_["dimensionality"] = {model_id: {"dimension": [], "accuracy": []} for model_id in self.model}
        assert hyperplane_dim+freq_projected_dim < max_projected_dim, "Number of noise dimensions should be larger than that of original dimensions."

        for prj_dim in range(hyperplane_dim, max_projected_dim+1, freq_projected_dim):
            start = time.time()
            
            sample_size = n_components * (cluster_samples_per_dim * prj_dim)
            Xs, ys, Ks = Robustness.gaussMixture_meanRandom_covWishart(sample_size, hyperplane_dim, n_components, d_proj=hyperplane_dim, 
                                                                       means=source_means, dim_noise_level=dim_noise_level)
            Xt, yt, Kt = Robustness.gaussMixture_meanRandom_covWishart(sample_size, hyperplane_dim, n_components, d_proj=hyperplane_dim, 
                                                                       means=target_means, dim_noise_level=dim_noise_level)
            K = [(Ks[i], Kt[i]) for i in range(len(Ks))][:n_keypoints]

            for model_id, model in self.model.items():
                acc = self.run(Xs, Xt, ys, yt, model, K=K)
                self.record_["dimensionality"][model_id]["dimension"].append(prj_dim)
                self.record_["dimensionality"][model_id]["accuracy"].append(acc)
            
            if (prj_dim - hyperplane_dim) % (2 * freq_projected_dim) == 0:
                info = {model_id: self.record_["dimensionality"][model_id]["accuracy"][-1] for model_id in self.record_["dimensionality"]}
                self.checkpoint()
                self.logger.info(f"Dimensions: {prj_dim}, Accuracy: {info}, Runtime: {time.time() - start}s")

        return self.record_["dimensionality"]
    
