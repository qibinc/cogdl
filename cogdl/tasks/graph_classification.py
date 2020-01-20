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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import shuffle as skshuffle
from tqdm import tqdm

from cogdl import options
from cogdl.data import Dataset
from cogdl.datasets import build_dataset
from cogdl.models import build_model

from . import BaseTask, register_task
from .unsupervised_node_classification import TopKRanker, UnsupervisedNodeClassification

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
        self.seed = args.seed

    def train(self):
        embeddings = self.model.train(None)

        # label nor multi-label
        label_matrix = self.label_matrix
        label_matrix = torch.Tensor(self.label_matrix)
        labels = np.array(label_matrix.argmax(axis=1).squeeze().tolist())

        self._evaluate = lambda x, y: self.svc_classify(x, y, False)
        # self._evaluate = lambda x, y: self.linearsvc_classify(x, y, False)
        # self._evaluate = lambda x, y: self.log_classify(x, y, False)
        # self._evaluate = lambda x, y: self.randomforest_classify(x, y, False)

        return self._evaluate(embeddings, labels)

    def log_classify(self, x, y, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        for train_index, test_index in kf.split(x, y):

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
            if search:
                # params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
                params = {"C": [1, 10, 100, 1000, 10000, 100000]}
                classifier = GridSearchCV(
                    LogisticRegression(multi_class="ovr", solver="liblinear"), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
                )
            else:
                classifier = LogisticRegression(C=100000, multi_class="ovr", solver="liblinear")
            classifier.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        return {"Micro-F1": np.mean(accuracies)}

    def svc_classify(self, x, y, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        for train_index, test_index in kf.split(x, y):

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
            if search:
                # params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
                params = {"C": [1, 10, 100, 1000, 10000, 100000]}
                classifier = GridSearchCV(
                    SVC(), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
                )
            else:
                classifier = SVC(C=100000)
            classifier.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        return {"Micro-F1": np.mean(accuracies)}

    def linearsvc_classify(self, x, y, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        for train_index, test_index in kf.split(x, y):

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if search:
                # params = {"C": [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]}
                params = {"C": [1, 10, 100, 1000, 10000, 100000]}
                classifier = GridSearchCV(
                    LinearSVC(), params, cv=5, scoring="accuracy", verbose=0, n_jobs=-1
                )
            else:
                classifier = LinearSVC(C=100000)
            classifier.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        return {"Micro-F1": np.mean(accuracies)}

    def randomforest_classify(self, x, y, search):
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.seed)
        accuracies = []
        for train_index, test_index in kf.split(x, y):

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if search:
                params = {"n_estimators": [100, 200, 500, 1000]}
                classifier = GridSearchCV(
                    RandomForestClassifier(n_jobs=-1),
                    params,
                    cv=5,
                    scoring="accuracy",
                    verbose=0,
                    n_jobs=-1
                )
            else:
                classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            classifier.fit(x_train, y_train)
            accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))
        return {"Micro-F1": np.mean(accuracies)}
