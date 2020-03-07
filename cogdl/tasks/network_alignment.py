import copy
import random
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from cogdl import options
from cogdl.data import Dataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task


@register_task("align")
class Align(BaseTask):
    """Network alignment task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=128)
        # fmt: on

    def __init__(self, args):
        super(Align, self).__init__(args)
        dataset = build_dataset(args)
        self.data = dataset[0]

        self.model = build_model(args)
        self.hidden_size = args.hidden_size
        self.args = args

    def _train_graph(self, data):
        G = nx.MultiGraph()
        G.add_edges_from(data.edge_index.t().tolist())
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((G.number_of_nodes(), self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]
        return features_matrix

    def train(self):
        emb_1 = self._train_graph(self.data[0])
        emb_2 = self._train_graph(self.data[1])

        return self._evaluate(emb_1, emb_2, self.data[0].y, self.data[1].y)

    def _evaluate(self, emb_1, emb_2, dict_1, dict_2):
        print(len(dict_1), len(dict_2))
        shared_keys = set(dict_1.keys()) & set(dict_2.keys())
        # HACK: deal with this later
        shared_keys = list(
            filter(
                lambda x: dict_1[x] < emb_1.shape[0] and dict_2[x] < emb_2.shape[0],
                shared_keys,
            )
        )
        print(f"num shared keys {len(shared_keys)}")
        emb_1 /= np.linalg.norm(emb_1, axis=1).reshape(-1, 1)
        emb_2 /= np.linalg.norm(emb_2, axis=1).reshape(-1, 1)
        reindex = [dict_2[key] for key in shared_keys]
        reindex_dict = dict([(x, i) for i, x in enumerate(reindex)])
        emb_2 = emb_2[reindex]
        # k_list = range(0, 101, 20)
        k_list = [1, 5, 10, 20]
        id2name = dict([(dict_2[k], k) for k in dict_2])

        all_results = defaultdict(list)
        for key in shared_keys:
            v = emb_1[dict_1[key]]
            scores = emb_2.dot(v)

            idxs = scores.argsort()[::-1]
            for k in k_list:
                all_results[k].append(int(reindex_dict[dict_2[key]] in idxs[:k]))
        res = dict(
            (f"Recall @ {k}", sum(all_results[k]) / len(all_results[k])) for k in k_list
        )

        plt.scatter(x=k_list, y=[res[f"Recall @ {k}"] for k in k_list])
        plt.xlim(0, 100)
        plt.ylim(0, 0.3)
        plt.savefig(self.args.dataset)

        return res
