import numpy as np
from scipy.stats import truncnorm
import torch
from torch import nn as nn


def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal_np(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values


def truncated_normal(size, std):
    x = std * truncated_normal_np(size, threshold=2)
    return torch.from_numpy(x).type(torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b