from .domain_adaptation import DomainAdaptation
from ..models._ot import _OT
from ..models import KeypointFOT

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


class AlphaSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name="mix_sensitive", log_dir=log_dir, classifier=classifier, **kwargs)

    def __call__(
        self,
        min_alpha: float = 0.0,
        max_alpha: float = 1.0,
        freq_alpha: float = 0.1,
        keypoints_per_cls: int = 1,
        n_keypoints: int = 10,
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"guide_mix": [], "accuracy": [], "runtime": []} for model_id in self.model}

        mnist_train_logits = np.array(self.classifier(self.mnist_X_train).detach())
        usps_test_logits = np.array(self.classifier(self.usps_X_test).detach())

        mnist_keypoints = DomainAdaptation.keypoints(mnist_train_logits, self.mnist_y_train.numpy(), keypoints_per_cls)
        usps_keypoints = DomainAdaptation.keypoints(usps_test_logits, self.usps_y_test.numpy(), keypoints_per_cls)
        K = [(usps_keypoints[i], mnist_keypoints[i]) for i in range(len(mnist_keypoints))]
        assert len(K) >= n_keypoints, f"Expected number of keypoints {n_keypoints} is greater than that of keypoints ({len(K)}) initialized."
        K = K[:n_keypoints]

        guide_mixes = np.arange(min_alpha, max_alpha+freq_alpha, freq_alpha)
        for guide_mix in guide_mixes:
            start = time.time()
            self.model["KeypointFOT"].alpha = guide_mix
            adapt_logits = self.run(usps_test_logits, mnist_train_logits, self.model["KeypointFOT"], K=K)


            self.record_[self.exp_name]["KeypointFOT"]["guide_mix"] \
                .append(guide_mix)
            self.record_[self.exp_name]["KeypointFOT"]["accuracy"] \
                .append(DomainAdaptation.accuracy(adapt_logits, self.usps_y_test.numpy()))
            self.record_[self.exp_name]["KeypointFOT"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{int(guide_mix*100)}% guiding] Accuracy: {self.record_[self.exp_name]["KeypointFOT"]["accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointFOT"]["runtime"][-1]}')

        self.plot(x_axis="guide_mix", y_axis="accuracy")
        return self.record_[self.exp_name]


class KeypointSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name="keypoint_sensitive", log_dir=log_dir, classifier=classifier, **kwargs)

    def  __call__(
        self,
        min_keypoints: int = 1,
        max_keypoints: Optional[int] = None,
        freq_keypoints: int = 1,
        keypoints_per_cls: int = 10,
    ) -> Dict:
        def sort_keypoints(K, keypoints_per_cls):
            K_ = []
            for i in range(keypoints_per_cls):
                K_ += [K[i + j * keypoints_per_cls] for j in range(len(K) // keypoints_per_cls)]
            return K_
        
        self.record_[self.exp_name] = {model_id: {"keypoints": [], "accuracy": [], "runtime": []} for model_id in self.model}

        mnist_train_logits = np.array(self.classifier(self.mnist_X_train).detach())
        usps_test_logits = np.array(self.classifier(self.usps_X_test).detach())

        mnist_keypoints = DomainAdaptation.keypoints(mnist_train_logits, self.mnist_y_train.numpy(), keypoints_per_cls)
        usps_keypoints = DomainAdaptation.keypoints(usps_test_logits, self.usps_y_test.numpy(), keypoints_per_cls)    
        K = [(usps_keypoints[i], mnist_keypoints[i]) for i in range(len(mnist_keypoints))]
        K = sort_keypoints(K, keypoints_per_cls)

        if max_keypoints is None:
            max_keypoints = len(mnist_keypoints)
        max_keypoints = min(max_keypoints, len(mnist_keypoints))

        n_keypoints = np.arange(min_keypoints, max_keypoints+1, freq_keypoints)
        for k in n_keypoints:
            k = int(k)
            start = time.time()
            adapt_logits = self.run(usps_test_logits, mnist_train_logits, self.model["KeypointFOT"], K=K[:k])

            self.record_[self.exp_name]["KeypointFOT"]["keypoints"] \
                .append(k)
            self.record_[self.exp_name]["KeypointFOT"]["accuracy"] \
                .append(DomainAdaptation.accuracy(adapt_logits, self.usps_y_test.numpy()))
            self.record_[self.exp_name]["KeypointFOT"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{k} keypoints] Accuracy: {self.record_[self.exp_name]["KeypointFOT"]["accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointFOT"]["runtime"][-1]}')

        self.plot(x_axis="keypoints", y_axis="accuracy")
        return self.record_[self.exp_name]
    

class EpsilonSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name="epsilon_sensitive", log_dir=log_dir, classifier=classifier, **kwargs)
    
    def __call__(
        self,
        eps_range: List = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
        keypoints_per_cls: int = 1,
        n_keypoints: int = 10,
    ):
        self.record_[self.exp_name] = {model_id: {"eps": [], "accuracy": [], "runtime": []} for model_id in self.model}

        mnist_train_logits = np.array(self.classifier(self.mnist_X_train).detach())
        usps_test_logits = np.array(self.classifier(self.usps_X_test).detach())

        mnist_keypoints = DomainAdaptation.keypoints(mnist_train_logits, self.mnist_y_train.numpy(), keypoints_per_cls)
        usps_keypoints = DomainAdaptation.keypoints(usps_test_logits, self.usps_y_test.numpy(), keypoints_per_cls)
        K = [(usps_keypoints[i], mnist_keypoints[i]) for i in range(len(mnist_keypoints))]
        assert len(K) >= n_keypoints, f"Expected number of keypoints {n_keypoints} is greater than that of keypoints ({len(K)}) initialized."
        K = K[:n_keypoints]

        for eps in eps_range:
            start = time.time()
            self.model["KeypointFOT"].eps = eps
            adapt_logits = self.run(usps_test_logits, mnist_train_logits, self.model["KeypointFOT"], K=K)

            self.record_[self.exp_name]["KeypointFOT"]["eps"] \
                .append(eps)
            self.record_[self.exp_name]["KeypointFOT"]["accuracy"] \
                .append(DomainAdaptation.accuracy(adapt_logits, self.usps_y_test.numpy()))
            self.record_[self.exp_name]["KeypointFOT"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{eps} entropy] Accuracy: {self.record_[self.exp_name]["KeypointFOT"]["accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointFOT"]["runtime"][-1]}')

        self.plot(x_axis="eps", y_axis="accuracy")
        return self.record_[self.exp_name]
    

class RhoSensitivity(DomainAdaptation):
    def __init__(
        self,
        model: Dict[str, _OT],
        log_dir: str,
        classifier: Optional[nn.Module] = None,
        **kwargs
    ):
        super().__init__(model, exp_name="rho_sensitive", log_dir=log_dir, classifier=classifier, **kwargs)

    def __call__(
        self,
        rho_range: List = [0.05, 0.07, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5],
        keypoints_per_cls: int = 1,
        n_keypoints: int = 10,
    ) -> Dict:
        self.record_[self.exp_name] = {model_id: {"rho": [], "accuracy": [], "runtime": []} for model_id in self.model}

        mnist_train_logits = np.array(self.classifier(self.mnist_X_train).detach())
        usps_test_logits = np.array(self.classifier(self.usps_X_test).detach())

        mnist_keypoints = DomainAdaptation.keypoints(mnist_train_logits, self.mnist_y_train.numpy(), keypoints_per_cls)
        usps_keypoints = DomainAdaptation.keypoints(usps_test_logits, self.usps_y_test.numpy(), keypoints_per_cls)
        K = [(usps_keypoints[i], mnist_keypoints[i]) for i in range(len(mnist_keypoints))]
        assert len(K) >= n_keypoints, f"Expected number of keypoints {n_keypoints} is greater than that of keypoints ({len(K)}) initialized."
        K = K[:n_keypoints]

        for rho in rho_range:
            start = time.time()
            self.model["KeypointFOT"].rho = rho
            adapt_logits = self.run(usps_test_logits, mnist_train_logits, self.model["KeypointFOT"], K=K)

            self.record_[self.exp_name]["KeypointFOT"]["rho"] \
                .append(rho)
            self.record_[self.exp_name]["KeypointFOT"]["accuracy"] \
                .append(DomainAdaptation.accuracy(adapt_logits, self.usps_y_test.numpy()))
            self.record_[self.exp_name]["KeypointFOT"]["runtime"] \
                .append(time.time() - start)
            self.checkpoint()
            self.logger.info(f'[{rho} temperature] Accuracy: {self.record_[self.exp_name]["KeypointFOT"]["accuracy"][-1]}, Runtime: {self.record_[self.exp_name]["KeypointFOT"]["runtime"][-1]}')

        self.plot(x_axis="rho", y_axis="accuracy")
        return self.record_[self.exp_name]
