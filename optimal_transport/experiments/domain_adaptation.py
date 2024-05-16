from ._experiment import Experiment
from ..models._ot import _OT
from ..utils import mnist_mlp

from typing import Dict, Optional, List, Tuple, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import os
import time
import matplotlib.pyplot as plt


class DomainAdaptationExperiment(Experiment):
    def __init__(
        self,
        model: Dict[str, _OT],
        exp_name: str,
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name, log_dir)

        (self.mnist_X_train, self.mnist_y_train), (self.mnist_X_test, self.mnist_y_test) = self.mnist(**kwargs)
        (self.usps_X_train, self.usps_y_train), (self.usps_X_test, self.usps_y_test) = self.usps(**kwargs)
        
        if classifier is None:
            self.classifier = mnist_mlp(pretrained=True).eval()
    
    def run(
        self, 
        xs: np.ndarray, xt: np.ndarray, a: np.ndarray, b: np.ndarray,
        model: _OT, 
        **kwargs
    ) -> np.ndarray:
        model.fit(xs, xt, a=a, b=b, **kwargs)
        return model.transport(xs, xt)

    def plot(
        self, x_axis: str, y_axis: str,
        save_fig: bool = True, **kwargs
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

    def _benchmark(self, classifier: nn.Module):
        self.record_["None"] = {}
        self.record_["None"]["mnist_train"] = self.accuracy(classifier(self.mnist_X_train).detach().numpy(), self.mnist_y_train.numpy())
        self.record_["None"]["mnist_test"] = self.accuracy(classifier(self.mnist_X_test).detach().numpy(), self.mnist_y_test.numpy())
        self.record_["None"]["usps_train"] = self.accuracy(classifier(self.usps_X_train).detach().numpy(), self.usps_y_train.numpy())
        self.record_["None"]["usps_test"] = self.accuracy(classifier(self.usps_X_test).detach().numpy(), self.usps_y_test.numpy())
        self.logger.info(f'Accuracy: {self.record_["None"]}')
        self.checkpoint()

    @property
    def classifier(self):
        return self._classifier
    
    @classifier.setter
    def classifier(self, net: nn.Module):
        self._classifier = net   
        self._benchmark(net)

    @classmethod
    def accuracy(
        cls, preds: np.ndarray, labels: np.ndarray
    ) -> float:
        y_pred = np.argmax(preds, axis=1)
        acc = (y_pred == labels).sum() / labels.shape[0]
        return acc
    
    @classmethod
    def keypoints(
        cls, X: np.ndarray, y: np.ndarray, 
        keypoints_per_cls: int = 1
    ) -> List:
        def euclidean(source, target, p=2):
            return np.sum(
                np.power(
                    source.reshape([source.shape[0], 1, source.shape[1]]) -
                    target.reshape([1, target.shape[0], target.shape[1]]),
                    p
                ),
                axis=-1
            ) ** 1/2
        
        labels = np.unique(y)
        selected_inds = []
        for label in labels:
            cls_indices = np.where(y == label)[0]
            distance = euclidean(X[cls_indices], np.mean(X[cls_indices], axis=0)[None, :]).squeeze()
            selected_inds.extend(cls_indices[np.argsort(distance)[:keypoints_per_cls]])
        return selected_inds
    
    @classmethod
    def mnist(
        cls, root_dir: str = "datasets",
        train_size: int = 1000, test_size: int = 2000, transform=None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        # load train set
        train_dataset = datasets.MNIST(root=os.path.join(root_dir, "mnist"), 
                                       train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)
        
        # load test set
        test_dataset = datasets.MNIST(root=os.path.join(root_dir, "mnist"), 
                                      train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False)

        return next(iter(train_loader)), next(iter(test_loader))

    @classmethod
    def usps(
        cls, root_dir: str = "datasets",
        train_size: int = 1000, test_size: int = 2000, transform=None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if transform is None:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Pad(6)])

        # load train set
        train_dataset = datasets.USPS(root=os.path.join(root_dir, "usps"), 
                                      train=True, download=True, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=train_size, shuffle=True)

        # load test set
        test_data = datasets.USPS(root=os.path.join(root_dir, "usps"), 
                                  train=False, download=True, transform=transform)
        test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)

        return next(iter(train_loader)), next(iter(test_loader))