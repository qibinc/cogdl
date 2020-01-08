import json
import os
import os.path as osp
import sys
from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np
import torch

from cogdl.data import Data, Dataset

from . import register_dataset


@register_dataset("align_single")
class AlignSingleDataset(Dataset):
    def __init__(self, root, name):
        edge_index = self._preprocess(root, name)
        self.data = Data(x=None, edge_index=edge_index)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        print(f"num_nodes: {num_nodes}")
        print(f"num_edges: {len(edge_list)}")

        return torch.LongTensor(edge_list).t()


@register_dataset("kdd")
class KDD(AlignSingleDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "kdd")


@register_dataset("icdm")
class ICDM(AlignSingleDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "icdm")


@register_dataset("sigir")
class SIGIR(AlignSingleDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "sigir")


@register_dataset("cikm")
class CIKM(AlignSingleDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "cikm")


@register_dataset("sigmod")
class SIGMOD(AlignSingleDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "sigmod")


@register_dataset("icde")
class ICDE(AlignSingleDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "icde")


@register_dataset("align")
class AlignDataset(Dataset):
    def __init__(self, root, name1, name2):
        edge_index_1, dict_1, self.node2id_1 = self._preprocess(root, name1)
        edge_index_2, dict_2, self.node2id_2 = self._preprocess(root, name2)
        self.data = [
            Data(x=None, edge_index=edge_index_1, y=dict_1),
            Data(x=None, edge_index=edge_index_2, y=dict_2),
        ]
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, root, name):
        dict_path = os.path.join(root, name + ".dict")
        graph_path = os.path.join(root, name + ".graph")

        with open(graph_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            f.readline()
            for line in f:
                x, y, t = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                # repeat t times
                for _ in range(t):
                    # to undirected
                    edge_list.append([node2id[x], node2id[y]])
                    edge_list.append([node2id[y], node2id[x]])

        name_dict = dict()
        with open(dict_path) as f:
            for line in f:
                name, str_x = line.split("\t")
                x = int(str_x)
                if x not in node2id:
                    node2id[x] = len(node2id)
                name_dict[name] = node2id[x]

        num_nodes = len(node2id)
        print(f"num_nodes: {num_nodes}")

        return torch.LongTensor(edge_list).t(), name_dict, node2id

@register_dataset("kdd_kdd")
class KDDICDM(AlignDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "kdd", "kdd")

@register_dataset("kdd_icdm")
class KDDICDM(AlignDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "kdd", "icdm")


@register_dataset("sigir_cikm")
class SIGIRCIKM(AlignDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "sigir", "cikm")


@register_dataset("sigmod_icde")
class KDDICDM(AlignDataset):
    def __init__(self):
        super().__init__("cogdl/data/panther/", "sigmod", "icde")
