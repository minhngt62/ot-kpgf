import numpy as np
from typing import List, Tuple
import torch
from sklearn.cluster import KMeans
from ot.bregman import sinkhorn

from ..utils import js_div, squared_euclidean, softmax


class KeypointFOT:
    def __init__(
        self,
        Xs: torch.Tensor, Xt: torch.Tensor, 
        a: torch.Tensor, b: torch.Tensor, 
        K: List[Tuple], k: List[Tuple],
        sinkhorn_reg: float = 0.001, temperature: float = 0.1, div_term: float = 1e-10, #guide_mixing: float = 0.95,
        stop_thr: float = 1e-7, max_iters: int = 1000
    ):
        self.X = Xs
        self.Y = Xt
        self.p = a
        self.q = b
        self.Z, self.h = self._init_anchors(k + len(K))

        if not self.p:
            self.p = torch.ones(len(self.X)) / len(self.X)
        if not self.q:
            self.q = torch.ones(len(self.Y)) / len(self.Y)

        self.I = torch.tensor([pair[0] for pair in K])
        self.J = torch.tensor([pair[1] for pair in K])
        self.L = torch.arange(len(self.I))

        self.eps = sinkhorn_reg
        self.rho = temperature
        self.div_term = div_term

        self.stop_thr = stop_thr
        self.max_iters = max_iters
        
        self.Mx, self.My = self._init_masks()

    def __call__(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for i in range(self.max_iters):
            Cx, Cy = self._compute_guide_costs()
            Px = self.update_plans(self.p, self.h, Cx, self.Mx)
            Py = self.update_plans(self.h, self.q, Cy, self.My)
            Z = self.update_anchors(Px, Py)

            err = torch.sqrt(torch.sum(torch.square(Z - self.Z), axis=1)).sum()
            if err <= self.stop_thr:
                print(f"Threshold reached at iteration {i}")
                break
            self.Z = Z

        return Z, Px, Py

    def update_anchors(
        self, 
        Px: torch.Tensor, 
        Py: torch.Tensor
    ) -> torch.Tensor:
        Z = 0.5 * (torch.matmul((Px).T, self.X) + torch.matmul(Py, self.Y)) * len(self.Z)
        return Z
    
    def update_plans(
        self,
        p: torch.Tensor, q: torch.Tensor,
        C: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        def M(u, v) -> torch.Tensor:
            "modified cost for logarithmic updates"
            "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
            M = (-C + torch.unsqueeze(u,1) + torch.unsqueeze(v,0)) / self.eps
            if mask is not None:
                M[mask==0] = -1e6
            return M

        def lse(A) -> torch.Tensor:
            "log-sum-exp"
            max_A, _ = torch.max(A, dim=1, keepdims=True)
            return torch.log(torch.exp(A-max_A).sum(1, keepdims=True) + self.div_term) + max_A  # add 10^-10 to prevent NaN

        # sinkhorn iteration ......................................................................
        u, v, err = 0. * p, 0. * q, 0.
        for _ in range(self.max_iters):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(p) - lse(M(u, v)).squeeze()) + u
            v = self.eps * (torch.log(q) - lse(M(u, v).T).squeeze()) + v

            err = torch.sum(torch.abs(u - u1))
            if err <= self.stop_thr:
                break
        
        U, V = u, v
        P = torch.exp(M(U, V))  # transport plan P = diag(u) * K * diag(v)
        return P
        

    def _init_anchors(
        self, 
        n_clusters: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        model = KMeans(n_clusters=n_clusters)
        model.fit(self.X)
        Z = model.cluster_centers_
        h = torch.ones(n_clusters) / (n_clusters)
        return torch.from_numpy(Z), h
    
    def _init_masks(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Mx = self.guide_mask(self.I, self.L, shape=(self.X.size(0), self.Z.size(0)))
        My = self.guide_mask(self.L, self.J, shape=(self.Z.size(0), self.Y.size(0)))
        return Mx, My
    
    def _compute_guide_costs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Gx = self.guide_matrix(self.X, self.Z, self.I, self.L, self.rho)
        Gy = self.guide_matrix(self.Z, self.Y, self.L, self.J, self.rho)
        return Gx, Gy
    
    def guide_mask(
        self, 
        I: torch.Tensor, J: torch.Tensor, 
        shape: Tuple
    ) -> torch.Tensor:
        mask = torch.ones(shape)
        mask[I, :] = 0
        mask[:, J] = 0
        mask[I, J] = 1
        return mask
    
    def guide_matrix(
        self,
        Xs: torch.Tensor, Xt: torch.Tensor,
        I: torch.Tensor, J: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        CXs = squared_euclidean(Xs, Xs)
        CXt = squared_euclidean(Xt, Xt)
        CXs_kp = CXs[:, I]
        CXt_kp = CXt[:, J]
        R1 = softmax(-2 * CXs_kp / temperature)
        R2 = softmax(-2 * CXt_kp / temperature)
        return js_div(R1, R2)


class KeypointKantorovichFOT(KeypointFOT):
    def __init__(
        self,
        Xs: np.ndarray, Xt: np.ndarray, 
        a: np.ndarray, b: np.ndarray, 
        K: List[Tuple], k: List[Tuple],
        sinkhorn_reg: float = 0.001, temperature: float = 0.1, div_term: float = 1e-10, guide_mixing: float = 0.95,
        stop_thr: float = 1e-7, max_iters: int = 1000
    ):
        super(KeypointFOT, self).__init__(Xs, Xt, a, b, K, k, 
                                          sinkhorn_reg, temperature, div_term, stop_thr, max_iters)
        self.alpha = guide_mixing
    
    def _compute_guide_costs(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Cx = squared_euclidean(self.X, self.Z)
        Cy = squared_euclidean(self.Z, self.Y)

        Gx = self.alpha * Cx + (1 - self.alpha) * self.guide_matrix(self.X, self.Z, self.I, self.L, self.rho)
        Gy = self.alpha * Cy + (1 - self.alpha) * self.guide_matrix(self.Z, self.Y, self.L, self.J, self.rho)
        return Gx, Gy 