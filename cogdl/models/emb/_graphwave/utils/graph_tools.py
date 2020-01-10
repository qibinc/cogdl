# -*- coding: utf-8 -*-
"""
Tools for the analysis of the Graph
"""
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sb
import scipy as sc


def laplacian(a):
        n_nodes, _ = a.shape
        posinv = np.vectorize(lambda x:  float(1.0)/np.sqrt(x) if x>1e-10 else 0.0)
        d = sc.sparse.diags(np.array(posinv(a.sum(0))).reshape([-1,]),0)
        lap = sc.sparse.eye(n_nodes) - d.dot(a.dot(d))
        return lap


def degree_matrix(adj):
    n, _ =adj.shape
    deg = np.diag([np.sum(adj[i, :]) for i in range(n)])
    return deg


def Invdegree_matrix(adj):
    n, _ = adj.shape
    pos = np.vectorize(lambda x: x if x > 0 else 1)
    deg_prov = pos(np.array(adj.sum(0)))
    deg = np.diag(1.0 / deg_prov)
    return deg


def normalize_matrix(m, direction="row", type_norm="max"):
    n, _ = m.shape
    if direction == "row":
        if type_norm == "max":
            deg = [1.0 / np.max(m[i, :]) for i in range(n)]
        elif type_norm == "l2":
            deg=[1.0 / np.linalg.norm(m[i, :]) for i in range(n)]
        elif type_norm == "l1":
            deg=[1.0 / np.sum(np.abs(m[i,:])) for i in range(n)]
        else:
            print("direction not recognized. degefaulting to l2")
            deg=[1.0 / np.linalg.norm(m[i, :]) for i in range(n)]
        deg = np.diag(deg)
        return deg.dot(m)
    elif direction == "column":
        m_tilde = normalize_matrix(m.T, direction="row", type_norm=type_norm)
        return m_tilde.T
    else:
        print("direction not recognized. degefaulting to column")
        return normalize_matrix(m.T, direction="row", type_norm=type_norm)
