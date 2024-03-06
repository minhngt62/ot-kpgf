from ._ot import _OT
from ..utils import Distance, SquaredEuclidean, JSDiv, softmax

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
        stop_thr: float = 1e-7, 
        max_iters: int = 1000
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
        self.z_: Optional[np.ndarray] = None
        

    def fit(
        self,
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        K: List[Tuple],
        **kwargs,
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

            err = np.sqrt(np.sum(np.square(z - self.z_), axis=1)).sum()
            self.z_ = z
            if err <= self.stop_thr:
                print(f"Threshold reached at iteration {i}")
                break
        
        self.Pa_ = Ps
        self.Pb_ = Pt
        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
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
        Ps: np.ndarray, 
        Pt: np.ndarray
    ) -> np.ndarray:
        assert self.z_ is not None, "_init_anchors() did not run properly."
        z = 0.5 * (np.matmul((Ps).T, xs) + np.matmul(Pt, xt)) * len(self.z_)
        return z
    
    def _update_plans(
        self,
        p: np.ndarray, q: np.ndarray,
        C: np.ndarray, mask: np.ndarray,
    ) -> np.ndarray:
        C /= C.max() # normalized
        
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
        actual_nits = 0  # to check if algorithm terminates because of threshold or max iterations reached

        for i in range(self.max_iters):
            u1 = u  # useful to check the update
            u = self.eps * (np.log(p) - lse(M(u, v)).squeeze()) + u
            v = self.eps * (np.log(q) - lse(M(u, v).T).squeeze()) + v
            err = np.linalg.norm(u - u1)
            actual_nits += 1
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
        Cx, Cy = Cx / (Cx.max() + self.div_term), Cy / (Cy.max() + self.div_term)

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
        CXs = self.dist_fn(xs, xs)
        CXt = self.dist_fn(xt, xt)
        CXs_kp = CXs[:, I]
        CXt_kp = CXt[:, J]
        R1 = softmax(-2 * CXs_kp / self.rho)
        R2 = softmax(-2 * CXt_kp / self.rho)
        G = self.sim_fn(R1, R2)
        return G / (G.max() + self.div_term)
    

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
        self.z_: Optional[np.ndarray] = None

    
    def fit(
        self, 
        xs: np.ndarray, 
        xt: np.ndarray,
        a: Optional[np.ndarray],
        b: Optional[np.ndarray],
        **kwargs,
    ) -> "FOT":
        z0 = self._init_anchors(xs, self.n_anchors)
        self.Pa_, self.Pb_, self.z_ = ot.factored.factored_optimal_transport(
            xs, xt, a, b,
            X0=z0,
            reg=self.eps,
            r=self.n_anchors,
            stopThr=self.stop_thr,
            numItermax=self.max_iters,
            method=self.sinkhorn_method
        )

        return self

    def transport(
        self,
        xs: np.ndarray,
        xt: np.ndarray,
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
    ) -> np.ndarray:
        model = KMeans(n_clusters=n_clusters)
        model.fit(x)
        Z = model.cluster_centers_
        return Z
    
    
# @deprecated: Adopt from https://github.com/nerdslab/latentOT/blob/main/python_demo/utils.py
class LOT(_OT):
    def __init__(self, distance: Optional[Distance], n_source_anchors, n_target_anchors, epsilon=1, epsilon_z=1, intensity=[10, 10, 10], floyditer=50,
                 tolratio=1e-7, norm=2, random_state=None):
        super().__init__(distance)
        self.n_source_anchors, self.n_target_anchors = n_source_anchors, n_target_anchors

        self.epsilon = epsilon
        self.epsilon_z = epsilon_z

        self.intensity = intensity
        self.niter = floyditer
        self.tolratio = tolratio
        self.p = norm

        self.random_state = random_state

    @classmethod
    def compute_kmeans_centroids(cls, X, **kwargs):
        kmeans = KMeans(**kwargs).fit(X)
        return kmeans.cluster_centers_
    
    @classmethod
    def compute_cost_matrix(cls, source, target, p=2):
        cost_matrix = np.sum(np.power(source.reshape([source.shape[0], 1, source.shape[1]]) -
                                  target.reshape([1, target.shape[0], target.shape[1]]),
                                  p), axis=-1)
        return cost_matrix / (cost_matrix.max() + 1e-10)

    def fit(self, source: np.ndarray, target: np.ndarray, a=None, b=None, **kwargs) -> np.ndarray:
        # centroid initialized by K-means
        Cx = LOT.compute_kmeans_centroids(source, n_clusters=self.n_source_anchors, random_state=self.random_state)
        Cy = LOT.compute_kmeans_centroids(target, n_clusters=self.n_target_anchors, random_state=self.random_state)
        # Px, Py initialized by K-means and one-sided OT
        n = source.shape[0]
        m = target.shape[0]
        mu = 1 / n * np.ones([n, 1])
        nu = 1 / m * np.ones([m, 1])
        cost_xy = LOT.compute_cost_matrix(source, target, p=self.p)
        P = np.zeros([n,m]) + 1 / n / m

        converrlist = np.zeros(self.niter) + np.inf
        for t in range(0, self.niter):
            
            # compute cost matrices
            cost_x = LOT.compute_cost_matrix(source, Cx, p=self.p)
            cost_z = LOT.compute_cost_matrix(Cx, Cy, p=self.p)
            cost_y = LOT.compute_cost_matrix(Cy, target, p=self.p)
            Kx = np.exp(-self.intensity[0] * cost_x / self.epsilon)
            Kz = np.exp(-self.intensity[1] * cost_z / self.epsilon_z)
            Ky = np.exp(-self.intensity[2] * cost_y / self.epsilon)
            
            Pt1 = P
            Px, Py, Pz, P = self.update_transport_plans(Kx, Kz, Ky)  # update trans. plan

            # check for convergence
            converr = LA.norm(P - Pt1) / LA.norm(Pt1)
            converrlist[t] = converr
            if converr < self.tolratio:
                break

            # update anchors
            if t < self.niter - 1:
                Cx, Cy = self.update_anchors(Px, Py, Pz, source, target)

        self.Cx, self.Cy = Cx, Cy
        self.Px_, self.Py_, self.Pz_, self.P_ = Px, Py, Pz, P

    def update_transport_plans(self, Kx, Kz, Ky, niter=100, tol=1e-20, epsilon=0, clip_val=np.inf, epsilon1 = 0):
        dimx = Kx.shape[0]
        dimy = Ky.shape[1]
        dimz1, dimz2 = Kz.shape

        mu = 1 / dimx * np.ones([dimx, 1])
        nu = 1 / dimy * np.ones([dimy, 1])

        ax = np.ones([dimx, 1])
        bx = np.ones([dimz1, 1])
        ay = np.ones([dimz2, 1])
        by = np.ones([dimy, 1])
        az = np.ones([dimz1, 1])
        bz = np.ones([dimz2, 1])
        wxz = np.ones([dimz1, 1])
        wzy = np.ones([dimz2, 1])
        for i in range(1, niter + 1):
            
            ax = np.exp(np.minimum(np.log(np.maximum(mu,epsilon1)) - np.log(np.maximum(Kx.dot(bx), epsilon1)), clip_val))
            err1x = LA.norm(bx * Kx.T.dot(ax) - wxz, ord=1)
            

            by = np.exp(np.minimum(np.log(np.maximum(nu,epsilon1)) - np.log(np.maximum(Ky.T.dot(ay), epsilon1)), clip_val))
            err2y = LA.norm(ay * (Ky.dot(by)) - wzy, ord=1)
            
               
            wxz = ((az * (Kz.dot(bz))) * (bx * (Kx.T.dot(ax)))) ** (1 / 2)
            bx = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log( np.maximum(Kx.T.dot(ax),epsilon)), clip_val))
            err2x = LA.norm(ax * (Kx.dot(bx)) - mu, ord=1)

            az = np.exp(np.minimum(np.log(np.maximum(wxz, epsilon)) - np.log(np.maximum(Kz.dot(bz), epsilon)), clip_val))
            err1z = LA.norm(bz * Kz.T.dot(az) - wzy, ord=1)
            wzy = ((ay * (Ky.dot(by))) * (bz * (Kz.T.dot(az)))) ** (1 / 2)
            bz = np.exp(np.minimum(np.log(np.maximum(wzy,epsilon)) - np.log(np.maximum(Kz.T.dot(az), epsilon)), clip_val))
            err2z = LA.norm(az * (Kz.dot(bz)) - wxz, ord=1)

            ay = np.exp(np.minimum(np.log(np.maximum(wzy, epsilon)) - np.log(np.maximum(Ky.dot(by), epsilon)), clip_val))
            err1y = LA.norm(by * Ky.T.dot(ay) - nu, ord=1)
            if max(err1x, err2x, err1z, err2z, err1y, err2y) < tol:
                break

        Px = np.diagflat(ax).dot(Kx.dot(np.diagflat(bx)))
        Pz = np.diagflat(az).dot(Kz.dot(np.diagflat(bz)))
        Py = np.diagflat(ay).dot(Ky.dot(np.diagflat(by)))
        const = 0
        z1 = Px.T.dot(np.ones([dimx, 1])) + const
        z2 = Py.dot(np.ones([dimy, 1])) + const
        P = np.dot(Px / z1.T, np.dot(Pz, Py / z2))
        return Px, Py, Pz, P

    def update_anchors(self, Px, Py, Pz, source, target):
        n = source.shape[0]
        m = target.shape[0]
        Px = self.intensity[0] * Px
        Pz = self.intensity[1] * Pz
        Py = self.intensity[2] * Py

        temp = np.concatenate((np.diagflat(Px.T.dot(np.ones([n, 1])) +
                                           Pz.dot(np.ones([self.n_target_anchors, 1]))), -Pz), axis=1)
        temp1 = np.concatenate((-Pz.T, np.diagflat(Py.dot(np.ones([m, 1])) +
                                                   Pz.T.dot(np.ones([self.n_source_anchors, 1])))), axis=1)
        temp = np.concatenate((temp, temp1), axis=0)
        sol = np.concatenate((source.T.dot(Px), target.T.dot(Py.T)), axis=1).dot(LA.inv(temp))
        Cx = sol[:, 0:self.n_source_anchors].T
        Cy = sol[:, self.n_source_anchors:self.n_source_anchors + self.n_target_anchors].T
        return Cx, Cy

    def transport(self, source, target, **kwargs) -> np.ndarray:
        n = source.shape[0]
        m = target.shape[0]
        Cx_lot = self.Px_.T.dot(source) / (self.Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
        Cy_lot = self.Py_.dot(target) / (self.Py_.dot(np.ones([m, 1])) + 10 ** -20)
        transported = source + np.dot(
            np.dot(
                self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]),
                self.Pz_ / np.sum(self.Pz_, axis=1).reshape([self.n_source_anchors, 1])
            ),
            Cy_lot) - np.dot(self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]), Cx_lot)
        return transported

    def robust_transport(self, source, target, threshold=0.8, decay=0) -> np.ndarray:
        n = source.shape[0]
        m = target.shape[0]
        Cx_lot = self.Px_.T.dot(source) / (self.Px_.T.dot(np.ones([n, 1])) + 10 ** -20)
        Cy_lot = self.Py_.dot(target) / (self.Py_.dot(np.ones([m, 1])) + 10 ** -20)

        maxPz = np.max(self.Pz_, axis=1)
        Pz_robust = self.Pz_.copy()

        for i in range(0, self.n_source_anchors):
            for j in range(0, self.n_target_anchors):
                if self.Pz_[i, j] < maxPz[i] * threshold:
                    Pz_robust[i, j] = self.Pz_[i, j] * decay
        Pz_robust = Pz_robust / np.sum(Pz_robust, axis=1).reshape([self.n_source_anchors, 1]) * \
                    np.sum(self.Pz_, axis=1).reshape([self.n_source_anchors, 1])

        transported = source + np.dot(
            np.dot(
                self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]),
                Pz_robust / np.sum(Pz_robust, axis=1).reshape([self.n_source_anchors, 1])
            ), Cy_lot) - np.dot(self.Px_ / np.sum(self.Px_, axis=1).reshape([n, 1]), Cx_lot)
        return transported