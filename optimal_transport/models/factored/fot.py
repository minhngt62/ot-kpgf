from .._ot import _OT
from ...utils import Distance, SquaredEuclidean, JSDiv, softmax

from typing import Optional, List, Tuple
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
import ot


class FOT(_OT):
    def __init__(
        self,
        n_anchors: int,
        distance: Distance = SquaredEuclidean,
        sinkhorn_reg: float = 0.001, 
        stop_thr: float = 1e-7, 
        max_iters: int = 1000,
        div_term: float = 1e-10,
        sinkhorn_method: str = "sinkhorn_log"
    ):
        super().__init__(distance)
        
        self.n_anchors = n_anchors
        self.eps = sinkhorn_reg
        self.stop_thr = stop_thr
        self.max_iters = max_iters
        self.div_term = div_term
        self.sinkhorn_method = sinkhorn_method

        self.Pa_: Optional[np.ndarray] = None
        self.Pb_: Optional[np.ndarray] = None
        self.P_: Optional[np.ndarray] = None
        self.z_: Optional[np.ndarray] = None

    
    def fit(
        self, 
        xs: np.ndarray, xt: np.ndarray, 
        a: Optional[np.ndarray], b: Optional[np.ndarray],
        **kwargs,
    ) -> "FOT":
        z0, h = self._init_anchors(xs, self.n_anchors)
        self.Pa_, self.Pb_, self.z_ = ot.factored.factored_optimal_transport(
            xs, xt, a, b, X0=z0,
            reg=self.eps,r=self.n_anchors, stopThr=self.stop_thr,
            numItermax=self.max_iters, method=self.sinkhorn_method
        )
        self.P_ = np.dot(
            self.Pa_ / (self.Pa_.T.dot(np.ones([xs.shape[0], 1]))).T, 
            self.Pb_)
        return self

    def transport(
        self,
        xs: np.ndarray, xt: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        n = xs.shape[0]
        m = xt.shape[0]
        assert (self.Pa_ is not None) and (self.Pb_ is not None), "Should run fit() before mapping"
        
        Cx = self.Pa_.T.dot(xs) / (self.Pa_.T.dot(np.ones((n, 1))) + self.div_term)
        Cy = self.Pb_.dot(xt) / (self.Pb_.dot(np.ones((m, 1))) + self.div_term)
        return xs + np.dot(self.Pa_ / np.sum(self.Pa_, axis=1).reshape([n, 1]), Cy - Cx)


    def _init_anchors(
        self, 
        x: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        model = KMeans(n_clusters=n_clusters)
        model.fit(x)
        Z = model.cluster_centers_
        h = np.ones(n_clusters) / (n_clusters)
        return Z, h