# -*- coding: utf-8 -*-
"""
Useful functions for investigating the distribution of the diffusion coefficients
"""
import copy
from .graph_tools import *
import numpy as np
import pandas as pd
import seaborn as sb


def h(x, epsilon=10**(-6)):
    if x > epsilon:
        return - (x) * np.log(x)
    elif x < 0:
        print("error: argument is negative")
        return np.nan
    else:
        return -(x + epsilon) * np.log(x + epsilon)


def entropy(mat, nb_bins=20):
    N, m = mat.shape
    ent = np.zeros(m)
    for i in range(m):
        h, w = np.histogram(mat[:, i], bins=nb_bins)
        v=[(0.5 * (w[k+1] - w[k]) + w[k]) for k in range(nb_bins)]
        ent[i] = 1.0 / N * np.sum([-h[k] * np.log(v[k])
                                   for k in range(nb_bins)])
    return ent


def variance_without_diagonal(mat, recompute_mean=False):
    var = np.zeros(mat.shape[1])
    mu = np.zeros(mat.shape[1])
    N = mat.shape[0]
    for i in range(mat.shape[1]):
        if recompute_mean:
            mu[i] = 1.0 / (N-1) * (N * np.mean(mat[:, i]) - mat[i, i])
        else:
            mu[i] = 1.0 / N
        var[i] = np.mean([(mat[j, i] - mu[i])**2
                          for j in range(mat.shape[0]) if j != i])
    return var, mu


def entropy_naive(mat, centered=False, offset=False, norm=False):
    ent = np.zeros(mat.shape[1])
    N = mat.shape[1]
    if centered and offset:
        centered = False
    if centered:
        mat2 = np.abs(mat - 1.0 / N * np.ones((mat.shape[0], mat.shape[1])))
    elif offset:
        mat2 = copy.deepcopy(mat)
        np.fill_diagonal(mat2, 0)
    else:
        mat2 = copy.deepcopy(mat)
    if norm:
        mat2 = normalize_matrix(mat2, direction="column", type_norm="l1")
    epsilon = 0.05 * 1.0 / N
    for i in range(mat.shape[1]):
        if offset:
            ent[i] = np.mean([- mat2[j, i] * np.log(mat2[j, i])
                             for j in range(mat.shape[0]) if j != i])
        else:
            ent[i] = np.mean([h(mat2[j, i], epsilon)
                              for j in range(mat.shape[0])])
    return ent
