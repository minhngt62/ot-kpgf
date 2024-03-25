import numpy as np
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from collections import OrderedDict
import torch

from ..classifiers import MLP

model_urls = {
    'mnist': 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/mnist-b07bb66b.pth'
}

def softmax(x: np.ndarray) -> np.ndarray:
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x - np.max(x)), axis=-1, keepdims=True)
    return f_x

def mnist_mlp(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=None) -> nn.Module:
    model = MLP(input_dims, n_hiddens, n_class)
    if pretrained is not None:
        m = model_zoo.load_url(model_urls['mnist'], map_location=torch.device('cpu'))
        state_dict = m.state_dict() if isinstance(m, nn.Module) else m
        assert isinstance(state_dict, (dict, OrderedDict)), type(state_dict)
        model.load_state_dict(state_dict)
    return model