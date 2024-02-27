import numpy as np
from scipy.stats import wishart
import scipy
from typing import Tuple, Optional, List, Union


def gaussMixture_meanRandom_covWishart(
    n: int, 
    d: int, 
    k: int,
    d_proj: Optional[int] = 5,
    means: Optional[np.ndarray] = None,
    covs: Optional[Union[np.ndarray, List]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert n % k == 0, "Each clusters should have equal number of samples."

    # Generate random means for each cluster
    if means is None:
        means = np.random.randn(k, d)
    
    # Sample covariance matrices from a Wishart distribution
    if covs is None:
        covs = []
        for _ in range(k):
            cov = scipy.stats.wishart.rvs(df=d, scale=np.eye(d))
            covs.append(cov)
    
    # Generate data points for each cluster
    X, y = [], []
    for i in range(k):
        Xi = np.random.multivariate_normal(means[i], covs[i], size=(n // k - 1))
        Xi = np.concatenate((means[i][None, :], Xi))
        X.append(Xi) # (n, d)
        y.append(i * np.ones(n // k))
    X, y = np.concatenate(X), np.concatenate(y)

    # Randomly project data points to a 5-dimensional subspace
    proj_mat = np.ones((d, d))
    if d_proj is not None:
        proj_mat = np.random.randn(d, d_proj)
    return np.dot(X, proj_mat), y, means