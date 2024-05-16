from ._experiment import Experiment
from ..models._ot import _OT
from ..classifiers import KNN

from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import scipy
import time
import matplotlib.pyplot as plt
import os
from numpy import linalg as LA


class RobustnessExperiment(Experiment):
    def __init__(
        self,
        model: Dict[int, _OT],
        exp_name: str,
        log_dir: str,
    ):
        super().__init__(model, exp_name, log_dir)

    def run(
        self, 
        xs: np.ndarray, xt: np.ndarray, a: np.ndarray, b: np.ndarray,
        ys: np.ndarray, yt: np.ndarray,
        model: _OT, n_neighbors: int = 1,
        **kwargs
    ) -> np.ndarray:
        knn = KNN(n_neighbors)
        knn.fit(xt, yt)
        model.fit(xs, xt, a, b, **kwargs)
        xs_t = model.transport(xs, xt)
        ys_pred = knn.predict(xs_t)
        return ys_pred

    def plot(
        self, x_axis: str, y_axis: str,
        save_fig: bool = False, 
        **kwargs
    ):
        plt.figure(figsize=(12, 8))
        for algo, record in self.record_[self.exp_name].items():
            plt.plot(record[x_axis], record[y_axis], label=algo)
        
        plt.title(self.exp_name)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend()
        plt.grid(True)

        if save_fig:
            plt.savefig(os.path.join(self.log_dir, f"{self.cur_time}.png"), dpi=300)
        plt.show()
    
    @classmethod
    def accuracy(
        cls, preds: np.ndarray, labels: np.ndarray
    ) -> float:
        return np.sum(preds == labels) / labels.shape[0]