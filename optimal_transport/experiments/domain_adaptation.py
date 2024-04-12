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


class DomainAdaptation(Experiment):
    def __init__(
        self,
        model: Dict[str, _OT],
        exp_name: str,
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name, log_dir)

        (self.mnist_X_train, self.mnist_y_train), (self.mnist_X_test, self.mnist_y_test) = DomainAdaptation.mnist(**kwargs)
        (self.usps_X_train, self.usps_y_train), (self.usps_X_test, self.usps_y_test) = DomainAdaptation.usps(**kwargs)
        
        if classifier is None:
            self.classifier = mnist_mlp(pretrained=True).eval()
    
    def _benchmark(self, classifier: nn.Module):
        self.record_["None"] = {}
        self.record_["None"]["mnist_train"] = DomainAdaptation.accuracy(classifier(self.mnist_X_train).detach().numpy(), self.mnist_y_train.numpy())
        self.record_["None"]["mnist_test"] = DomainAdaptation.accuracy(classifier(self.mnist_X_test).detach().numpy(), self.mnist_y_test.numpy())
        self.record_["None"]["usps_train"] = DomainAdaptation.accuracy(classifier(self.usps_X_train).detach().numpy(), self.usps_y_train.numpy())
        self.record_["None"]["usps_test"] = DomainAdaptation.accuracy(classifier(self.usps_X_test).detach().numpy(), self.usps_y_test.numpy())
        self.logger.info(f'Accuracy: {self.record_["None"]}')
        self.checkpoint()

    @property
    def classifier(self):
        return self._classifier
    
    @classifier.setter
    def classifier(self, net: nn.Module):
        self._classifier = net   
        self._benchmark(net)
    
    def run(
        self, xs: np.ndarray, xt: np.ndarray, model: _OT, **kwargs
    ) -> np.ndarray:
        n, n_ = xs.shape[0], xt.shape[0]
        model.fit(xs, xt, a=1/n*np.ones(n), b=1/n_*np.ones(n_), **kwargs)
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

    @classmethod
    def mnist(
        cls, root_dir: str = "datasets", seed: int = 5,
        train_size: int = 1000, test_size: int = 2000, transform=None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        torch.manual_seed(seed)
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
        cls, root_dir: str = "datasets", seed: int = 5,
        train_size: int = 1000, test_size: int = 2000, transform=None,
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        torch.manual_seed(seed)
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
    
    @classmethod
    def accuracy(
        cls, y_hat: np.ndarray, y: np.ndarray
    ) -> float:
        y_pred = np.argmax(y_hat, axis=1)
        acc = (y_pred == y).sum() / y.shape[0]
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
    

class USPSToMNIST(DomainAdaptation):
    def __init__(
        self, 
        model: Dict[str, _OT],
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name="usps_to_mnist", log_dir=log_dir, classifier=classifier, **kwargs)

    def __call__(
        self,
        keypoints_per_cls: int = 1,
        n_keypoints: int = 10
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"accuracy": [], "runtime": []} for model_id in self.model}
        
        mnist_train_logits = np.array(self.classifier(self.mnist_X_train).detach())
        usps_test_logits = np.array(self.classifier(self.usps_X_test).detach())

        mnist_keypoints = DomainAdaptation.keypoints(mnist_train_logits, self.mnist_y_train.numpy(), keypoints_per_cls)
        usps_keypoints = DomainAdaptation.keypoints(usps_test_logits, self.usps_y_test.numpy(), keypoints_per_cls)
        K = [(usps_keypoints[i], mnist_keypoints[i]) for i in range(len(mnist_keypoints))][:n_keypoints]

        for model_id in self.model:
            start = time.time()
            adapt_logits = self.run(usps_test_logits, mnist_train_logits, self.model[model_id], K=K)

            self.record_[self.exp_name][model_id]["accuracy"] \
                .append(DomainAdaptation.accuracy(adapt_logits, self.usps_y_test.numpy()))
            self.record_[self.exp_name][model_id]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{model_id}] Accuracy: {self.record_[self.exp_name][model_id]["accuracy"][0]}, Runtime: {self.record_[self.exp_name][model_id]["runtime"][0]}')

        return self.record_[self.exp_name]


class DUMNIST(DomainAdaptation):
    def __init__(
        self, 
        model: Dict[str, _OT],
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name="du_mnist", log_dir=log_dir, classifier=classifier, **kwargs)

    def __call__(
        self,
        keypoints_per_cls: int = 1,
        n_keypoints: int = 7,
        dropout_cls: List[int] = [0, 2, 8],
        size: int = 1000
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"accuracy": [], "runtime": [], "l2_error": []} for model_id in self.model}
        
        mnist_X_test, mnist_y_test = self._dropout_cls(classes=dropout_cls, size=size)
        mnist_test_logits_unb = self.classifier(mnist_X_test).detach().numpy()
        mnist_X_test = self._random_masking(mnist_X_test)
        mnist_train_logits = self.classifier(self.mnist_X_train).detach().numpy()
        mnist_test_logits_aug = self.classifier(mnist_X_test).detach().numpy()

        self.record_["None"]["mnist_test_aug"] \
            = DomainAdaptation.accuracy(mnist_test_logits_aug, mnist_y_test.numpy())
        self.logger.info(f'[None] Accuracy: {self.record_["None"]["mnist_test_aug"]}')

        aug_keypoints = DomainAdaptation.keypoints(mnist_test_logits_aug, mnist_y_test.numpy(), keypoints_per_cls)
        train_keypoints = torch.tensor(
            DomainAdaptation.keypoints(mnist_train_logits, self.mnist_y_train.numpy(), 
                                       keypoints_per_cls))[mnist_y_test.unique()].tolist()
        K = [(aug_keypoints[i], train_keypoints[i]) for i in range(len(aug_keypoints))][:n_keypoints]

        for model_id in self.model:
            start = time.time()
            adapt_logit = self.run(mnist_test_logits_aug, mnist_train_logits, self.model[model_id], K=K)
            
            self.record_[self.exp_name][model_id]["accuracy"].append(DomainAdaptation.accuracy(adapt_logit, mnist_y_test.numpy()))
            self.record_[self.exp_name][model_id]["l2_error"].append(np.linalg.norm(adapt_logit - mnist_test_logits_unb))
            self.record_[self.exp_name][model_id]["runtime"].append(time.time() - start)
            score = self.record_[self.exp_name][model_id]["accuracy"][0]
            err = self.record_[self.exp_name][model_id]["l2_error"][0]
            runtime = self.record_[self.exp_name][model_id]["runtime"][0]
            self.logger.info(f"[{model_id}] Accuracy: {score}, L2: {err}, Runtime: {runtime}")
            self.checkpoint()

        return self.record_[self.exp_name]


    def _dropout_cls(
        self, classes: List[int], size: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ones_like(self.mnist_y_test, dtype=torch.bool)
        for idx in classes:
            mask = torch.logical_and(mask, self.mnist_y_test != idx)
        return self.mnist_X_test[mask][:size], self.mnist_y_test[mask][:size]
    
    def _random_masking(
        self, X_unbalance: torch.Tensor
    ) -> torch.Tensor:
        mask = np.ones_like(X_unbalance[0])
        mask[0, :10, :10] = 0; mask[0, :10, -9:] = 0; mask[0, -9:, -9:] = 0; mask[0, 10:-9, 10:-9] = 0
        mnist_X_test_aug = torch.Tensor(X_unbalance * mask)
        return mnist_X_test_aug
    

class RobustSampling(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name="robust_sampling", log_dir=log_dir, classifier=classifier, **kwargs)
    
    def __call__(
        self,
        keypoints_per_cls: int = 1,
        n_keypoints: int = 10,
        min_samples: int = 50,
        max_samples: int = 1000,
        freq_samples: int = 50
    ):
        self.record_[self.exp_name] = {model_id: {"samples": [], "accuracy": [], "runtime": []} for model_id in self.model}

        mnist_train_logits = np.array(self.classifier(self.mnist_X_train).detach())
        usps_test_logits = np.array(self.classifier(self.usps_X_test).detach())

        inds = np.arange(len(mnist_train_logits))
        np.random.shuffle(inds)
        i = 0
        for n_samples in range(min_samples, max_samples+1, freq_samples):
            self.logger.info(f"------ At {n_samples} samples ------")
            inds_exp = inds[:n_samples]
            mnist_keypoints_ = DomainAdaptation.keypoints(mnist_train_logits[inds_exp], self.mnist_y_train.numpy()[inds_exp], keypoints_per_cls)
            usps_keypoints = torch.tensor(DomainAdaptation.keypoints(usps_test_logits, self.usps_y_test.numpy(), keypoints_per_cls)) \
                                                                     [self.mnist_y_train[inds_exp].unique()].tolist()
            K = [(usps_keypoints[i], mnist_keypoints_[i]) for i in range(len(usps_keypoints))][:n_keypoints]

            for model_id in self.model:
                start = time.time()
                adapt_logits = self.run(usps_test_logits, mnist_train_logits[inds_exp], self.model[model_id], K=K)
                
                self.record_[self.exp_name][model_id]["accuracy"].append(DomainAdaptation.accuracy(adapt_logits, self.usps_y_test.numpy()))
                self.record_[self.exp_name][model_id]["runtime"].append(time.time() - start)
                self.record_[self.exp_name][model_id]["samples"].append(n_samples)
                score = self.record_[self.exp_name][model_id]["accuracy"][i]
                runtime = self.record_[self.exp_name][model_id]["runtime"][i]
                self.logger.info(f"[{model_id}] Accuracy: {score}, Runtime: {runtime}")
                
            self.checkpoint()
            i += 1

        self.plot(x_axis="samples", y_axis="accuracy")
        return self.record_[self.exp_name]
