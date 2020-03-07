import copy
import random
import warnings
from collections import defaultdict

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from cogdl import options
from cogdl.data import Dataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task

warnings.filterwarnings("ignore")


@register_task("unsupervised_node_classification")
class UnsupervisedNodeClassification(BaseTask):
    """Node classification task."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--hidden-size", type=int, default=64)
        parser.add_argument("--num-shuffle", type=int, default=10)
        # fmt: on

    def __init__(self, args):
        super(UnsupervisedNodeClassification, self).__init__(args)
        dataset = build_dataset(args)
        self.data = dataset[0]
        try:
            import torch_geometric
        except ImportError:
            pyg = False
        else:
            pyg = True
        if pyg and issubclass(
            dataset.__class__.__bases__[0], torch_geometric.data.Dataset
        ):
            self.num_nodes = self.data.y.shape[0]
            self.num_classes = dataset.num_classes
            self.label_matrix = np.zeros((self.num_nodes, self.num_classes), dtype=int)
            self.label_matrix[range(self.num_nodes), self.data.y] = 1
        else:
            self.label_matrix = self.data.y
            self.num_nodes, self.num_classes = self.data.y.shape

        self.model = build_model(args)
        self.hidden_size = args.hidden_size
        self.num_shuffle = args.num_shuffle
        # self.is_weighted = self.data.edge_attr is not None
        self.is_weighted = None
        self.seed = args.seed

    def train(self):
        G = nx.Graph()
        if self.is_weighted:
            edges, weight = (
                self.data.edge_index.t().tolist(),
                self.data.edge_attr.tolist(),
            )
            G.add_weighted_edges_from(
                [(edges[i][0], edges[i][1], weight[0][i]) for i in range(len(edges))]
            )
        else:
            G.add_edges_from(self.data.edge_index.t().tolist())
        embeddings = self.model.train(G)

        # Map node2id
        features_matrix = np.zeros((self.num_nodes, self.hidden_size))
        for vid, node in enumerate(G.nodes()):
            features_matrix[node] = embeddings[vid]

        # label nor multi-label
        # label_matrix = sp.csr_matrix(self.label_matrix)
        label_matrix = torch.Tensor(self.label_matrix)

        return self._evaluate(features_matrix, label_matrix, self.num_shuffle)

    def _evaluate(self, features_matrix, label_matrix, num_shuffle):
        # features_matrix, node2id = utils.load_embeddings(args.emb)
        # label_matrix = utils.load_labels(args.label, node2id, divi_str=" ")

        # shuffle, to create train/test groups
        # shuffles = []
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = self.seed)
        idx_list = []
        labels = label_matrix.argmax(axis=1).squeeze().tolist()
        for idx in skf.split(np.zeros(len(labels)), labels):
            idx_list.append(idx)
        # for _ in range(num_shuffle):
        #     shuffles.append(skshuffle(features_matrix, label_matrix))

        # score each train/test group
        all_results = defaultdict(list)
        # training_percents = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # training_percents = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9]
        # training_percents = [0.8]

        # for train_percent in training_percents:
        for train_idx, test_idx in idx_list:

            X_train = features_matrix[train_idx]
            y_train = label_matrix[train_idx]

            X_test = features_matrix[test_idx]
            y_test = label_matrix[test_idx]

            clf = TopKRanker(LogisticRegression(C=1000))
            clf.fit(X_train, y_train)

            # find out how many labels should be predicted
            top_k_list = y_test.sum(axis=1).long().tolist()
            preds = clf.predict(X_test, top_k_list)
            result = f1_score(y_test, preds, average="micro")
            all_results[""].append(result)
        # print("micro", result)

        return dict(
            (
                f"Micro-F1 {train_percent}",
                sum(all_results[train_percent]) / len(all_results[train_percent]),
            )
            for train_percent in sorted(all_results.keys())
        )


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = np.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = sp.lil_matrix(probs.shape)

        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            for label in labels:
                all_labels[i, label] = 1
        return all_labels
