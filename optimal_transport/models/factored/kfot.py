from .._ot import _OT
from ...utils import Distance, SquaredEuclidean, JSDiv, softmax

from typing import Optional, List, Tuple
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
import ot


class KeypointFOT(_OT):
    def __init__(
        self,
        distance: Distance = SquaredEuclidean,
        similarity: Distance = JSDiv,
        n_free_anchors: Optional[int] = None,
        sinkhorn_reg: float = 0.004, 
        temperature: float = 0.1, 
        div_term: float = 1e-10, 
        guide_mixing: float = 0.6,
        stop_thr: float = 1e-5, 
        max_iters: int = 100
    ):
        super().__init__(distance)
        self.sim_fn = similarity

        self.k = n_free_anchors
        self.eps = sinkhorn_reg
        self.rho = temperature
        self.div_term = div_term
        self.stop_thr = stop_thr
        self.max_iters = max_iters
        self.alpha = 1 - guide_mixing

        self.Pa_: Optional[np.ndarray] = None
        self.Pb_: Optional[np.ndarray] = None
        self.P_: Optional[np.ndarray] = None
        self.z_: Optional[np.ndarray] = None
        

    def fit(
        self,
        xs: np.ndarray, xt: np.ndarray, 
        a: Optional[np.ndarray], b: Optional[np.ndarray], 
        K: List[Tuple], **kwargs,
    ) -> "KeypointFOT":
        z, h = self._init_anchors(xs, self.k + len(K))
        I, L, J = self._init_keypoint_inds(K)
        Ms, Mt = self._init_masks(xs, z, xt, I, L, J)

        self.z_ = z
        for i in range(self.max_iters):
            Cs, Ct = self._compute_cost_matrices(xs, xt, z, I, L, J)
            Ps = self._update_plans(a, h, Cs, Ms)
            Pt = self._update_plans(h, b, Ct, Mt)
            z = self._update_anchors(xs, xt, Ps, Pt)

            err = np.linalg.norm(z - self.z_)
            self.z_ = z
            if err <= self.stop_thr:
                #print(f"Threshold reached at iteration {i}")
                break
        
        self.I_, self.J_, self.L_ = I, J, L
        self.Pa_ = Ms * Ps
        self.Pb_ = Mt * Pt
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

    def _init_keypoint_inds(
        self,
        K: List[Tuple]
    ) -> Tuple[np.ndarray]:
        I = np.array([pair[0] for pair in K])
        J = np.array([pair[1] for pair in K])
        L = np.arange(len(K))
        return I, L, J

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
    
    def _init_masks(
        self,
        xs: np.ndarray, z: np.ndarray, xt: np.ndarray,
        I: np.ndarray, L: np.ndarray, J: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Ms = self._guide_mask(xs, z, I, L)
        Mt = self._guide_mask(z, xt, L, J)
        return Ms, Mt
    
    def _update_anchors(
        self, 
        xs: np.ndarray, xt: np.ndarray,
        Ps: np.ndarray, Pt: np.ndarray
    ) -> np.ndarray:
        assert self.z_ is not None, "_init_anchors() did not run properly."
        z = 0.5 * (np.matmul((Ps).T, xs) + np.matmul(Pt, xt)) * len(self.z_)
        return z
    
    def _update_plans(
        self,
        p: np.ndarray, q: np.ndarray,
        C: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        C = C / (C.max() + self.div_term)

        def M(u, v):
            "Modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            M =  (-C + np.expand_dims(u,1) + np.expand_dims(v,0)) / self.eps
            if mask is not None:
                M[mask==0] = -1e6
            return M

        def lse(A):
            "log-sum-exp"
            max_A = np.max(A, axis=1, keepdims=True)
            return np.log(np.exp(A-max_A).sum(1, keepdims=True) + self.div_term) + max_A  # add 10^-6 to prevent NaN

        # Actual Sinkhorn loop ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        for i in range(self.max_iters):
            u1 = u  # useful to check the update
            u = self.eps * (np.log(p) - lse(M(u, v)).squeeze()) + u
            v = self.eps * (np.log(q) - lse(M(u, v).T).squeeze()) + v
            err = np.linalg.norm(u - u1)
            if err < self.stop_thr:
                break

        U, V = u, v
        P = np.exp(M(U, V))  # P = diag(a) * K * diag(b)
        return P
    
    def _compute_cost_matrices(
        self,
        xs: np.ndarray, xt: np.ndarray, z: np.ndarray, 
        I: np.ndarray, L: np.ndarray, J: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        Cx = self.dist_fn(xs, z)
        Cy = self.dist_fn(z, xt)
        Cx = Cx / (Cx.max() + self.div_term)
        Cy = Cy / (Cy.max() + self.div_term)

        Gx = self.alpha * Cx + (1 - self.alpha) * self._guide_matrix(xs, z, I, L)
        Gy = self.alpha * Cy + (1 - self.alpha) * self._guide_matrix(z, xt, L, J)
        
        return Gx, Gy 

    def _guide_mask(
        self,
        xs: np.ndarray, xt: np.ndarray,
        I: np.ndarray, J: np.ndarray
    ) -> np.ndarray:
        mask = np.ones((xs.shape[0], xt.shape[0]))
        mask[I, :] = 0
        mask[:, J] = 0
        mask[I, J] = 1
        return mask

    def _guide_matrix(
        self,
        xs: np.ndarray, xt: np.ndarray,
        I: np.ndarray, J: np.ndarray,
    ) -> np.ndarray:
        Cs = self.dist_fn(xs, xs)
        Ct = self.dist_fn(xt, xt)
        Cs = Cs / (Cs.max() + self.div_term)
        Ct = Ct / (Ct.max() + self.div_term)

        Cs_kp = Cs[:, I]
        Ct_kp = Ct[:, J]
        R1 = softmax(-2 * Cs_kp / self.rho)
        R2 = softmax(-2 * Ct_kp / self.rho)
        G = self.sim_fn(R1, R2, eps=self.div_term)
        return G