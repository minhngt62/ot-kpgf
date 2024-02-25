import numpy as np
from scipy.stats import wishart
import scipy


def gaussMixture_meanRandom_covWishart(
    n: int, 
    d: int, 
    k: int,
    d_proj: int = 5,
    random_state: int = 0
) -> np.ndarray:
    assert n % k == 0, "Each clusters should have equal number of samples."

    # Generate random means for each cluster
    means = np.random.randn((k, d), random_state=random_state)
    
    # Sample covariance matrices from a Wishart distribution
    covs = []
    for _ in range(k):
        cov = scipy.stats.wishart.rvs(df=d, scale=np.eye(d), random_state=random_state)
        covs.append(cov)
    
    # Generate data points for each cluster
    X, y = [], []
    for i in range(k):
        X.append(np.random.multivariate_normal(means[i], covs[i], size=n, random_state=random_state)) # (n, d)
        y.append(i * np.ones(n // k))
    X, y = np.concatenate(X), np.concatenate(y)

    # Randomly project data points to a 5-dimensional subspace
    proj_mat = np.random.randn((k, d_proj), random_state=random_state)
    return np.dot(X, proj_mat)