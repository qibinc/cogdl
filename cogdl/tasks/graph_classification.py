
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from cogdl import options
from cogdl.data import Dataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task
from .unsupervised_node_classification import UnsupervisedNodeClassification

warnings.filterwarnings("ignore")

@register_task("graph_classification")
class GraphClassification(UnsupervisedNodeClassification):
    # HACK: Wrap graph classification as node classification

    def __init__(self, args):
        super(UnsupervisedNodeClassification, self).__init__(args)
        dataset = build_dataset(args)
        self.data = dataset.data
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

    def train(self):
        embeddings = self.model.train(None)

        # label nor multi-label
        label_matrix = sp.csr_matrix(self.label_matrix)

        return self._evaluate(embeddings, label_matrix, self.num_shuffle)
