import json
import os
import os.path as osp
import sys
from collections import defaultdict
from itertools import product

import networkx as nx
import numpy as np
import torch

from cogdl.data import Data, Dataset, download_url

from . import register_dataset


def read_edgelist_label_data(folder, prefix):
    graph_path = osp.join(folder, "{}.ungraph".format(prefix))
    cmty_path = osp.join(folder, "{}.cmty".format(prefix))

    G = nx.read_edgelist(graph_path, nodetype=int, create_using=nx.Graph())
    num_node = G.number_of_nodes()
    print("edge number: ", num_node)
    with open(graph_path) as f:
        context = f.readlines()
        print("edge number: ", len(context))
        edge_index = np.zeros((2, len(context)))
        for i, line in enumerate(context):
            edge_index[:, i] = list(map(int, line.strip().split("\t")))
    edge_index = torch.from_numpy(edge_index).to(torch.int)

    with open(cmty_path) as f:
        context = f.readlines()
        print("class number: ", len(context))
        label = np.zeros((num_node, len(context)))

        for i, line in enumerate(context):
            line = map(int, line.strip().split("\t"))
            for node in line:
                label[node, i] = 1

    y = torch.from_numpy(label).to(torch.float)
    data = Data(x=None, edge_index=edge_index, y=y)

    return data


class EdgelistLabel(Dataset):
    r"""networks from the https://github.com/THUDM/ProNE/raw/master/data

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset (:obj:`"Wikipedia"`).
    """

    url = "https://github.com/THUDM/ProNE/raw/master/data"

    def __init__(self, root, name):
        self.name = name
        super(EdgelistLabel, self).__init__(root)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        splits = [self.name]
        files = ["ungraph", "cmty"]
        return ["{}.{}".format(s, f) for s, f in product(splits, files)]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def get(self, idx):
        assert idx == 0
        return self.data

    def download(self):
        for name in self.raw_file_names:
            download_url("{}/{}".format(self.url, name), self.raw_dir)

    def process(self):
        data = read_edgelist_label_data(self.raw_dir, self.name)
        torch.save(data, self.processed_paths[0])


@register_dataset("dblp")
class DBLP(EdgelistLabel):
    def __init__(self):
        dataset = "dblp"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(DBLP, self).__init__(path, dataset)


class Edgelist(Dataset):
    def __init__(self, root, name):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        node_label_path = os.path.join(root, name + ".nodelabel")
        edge_index, y, self.node2id = self._preprocess(edge_list_path, node_label_path)
        self.data = Data(x=None, edge_index=edge_index, y=y)
        self.transform = None

    def get(self, idx):
        assert idx == 0
        return self.data

    def _preprocess(self, edge_list_path, node_label_path):
        with open(edge_list_path) as f:
            edge_list = []
            node2id = defaultdict(int)
            for line in f:
                x, y = list(map(int, line.split()))
                # Reindex
                if x not in node2id:
                    node2id[x] = len(node2id)
                if y not in node2id:
                    node2id[y] = len(node2id)
                edge_list.append([node2id[x], node2id[y]])
                edge_list.append([node2id[y], node2id[x]])

        num_nodes = len(node2id)
        with open(node_label_path) as f:
            nodes = []
            labels = []
            label2id = defaultdict(int)
            for line in f:
                x, label = list(map(int, line.split()))
                if label not in label2id:
                    label2id[label] = len(label2id)
                nodes.append(node2id[x])
                if "hindex" in self.name:
                    labels.append(label)
                else:
                    labels.append(label2id[label])
            if "hindex" in self.name:
                median = np.median(labels)
                labels = [int(label > median) for label in labels]
        assert num_nodes == len(set(nodes))
        y = torch.zeros(num_nodes, len(label2id))
        y[nodes, labels] = 1
        return torch.LongTensor(edge_list).t(), y, node2id


@register_dataset("usa_airport")
class USAAirport(Edgelist):
    def __init__(self):
        super().__init__("cogdl/data/struc2vec/", "usa-airports")


@register_dataset("brazil_airport")
class BrazilAirport(Edgelist):
    def __init__(self):
        super().__init__("cogdl/data/struc2vec/", "brazil-airports")


@register_dataset("europe_airport")
class EuropeAirport(Edgelist):
    def __init__(self):
        super().__init__("cogdl/data/struc2vec/", "europe-airports")

@register_dataset("h-index-rand-1")
class HIndexRand1(Edgelist):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_rand1_5000")

@register_dataset("h-index-top-1")
class HIndexTop1(Edgelist):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_top1_5000")

@register_dataset("h-index-rand20intop200")
class HIndexTop200(Edgelist):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_rand20intop200_5000")

class EdgelistWithAttr(Edgelist):
    def __init__(self, root, name):
        self.name = name
        edge_list_path = os.path.join(root, name + ".edgelist")
        node_label_path = os.path.join(root, name + ".nodelabel")
        node_features_path = os.path.join(root, name + ".nodefeatures")
        edge_index, y, self.node2id = self._preprocess(edge_list_path, node_label_path)
        self.num_classes = y.shape[1]
        x = self._preprocess_feats(node_features_path, self.node2id)
        assert x.shape[0] == y.shape[0] and x.shape[0] == len(self.node2id)
        y = y.argmax(dim=1)
        self.data = Data(x=x, edge_index=edge_index, y=y)
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        self.data.train_mask = torch.from_numpy(indices < int(0.2 * x.shape[0]))
        self.data.val_mask = torch.from_numpy((int(0.2 * x.shape[0]) <= indices) & (indices < int(0.3 * x.shape[0])))
        self.data.test_mask = torch.from_numpy(int(0.3 * x.shape[0]) <= indices)
        import torch_geometric.transforms as T
        self.transform = T.TargetIndegree()
    
    def _preprocess_feats(self, node_features_path, node2id):

        with open(node_features_path) as f:
            ids = []
            feats = []
            for line in f:
                line = list(map(float, line.split()))
                node = int(line[0])
                feat = line[1:]
                ids.append(node2id[node])
                feats.append(torch.FloatTensor(feat))
            feats = torch.stack(feats)
            feats[ids] = feats
        return feats

@register_dataset("h-index-rand-1-attr")
class HIndexRand1Attr(EdgelistWithAttr):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_rand1_5000")

@register_dataset("h-index-rand-1-attr-cat-struc")
class HIndexRand1Attr(EdgelistWithAttr):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_rand1_5000")
        struc_feat = np.load("saved/h-index-rand-1.npy")
        self.data.x = torch.cat([self.data.x, torch.from_numpy(struc_feat)], dim=1)

@register_dataset("h-index-top-1-attr")
class HIndexTop1Attr(EdgelistWithAttr):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_top1_5000")

@register_dataset("h-index-top-1-attr-cat-struc")
class HIndexTop1Attr(EdgelistWithAttr):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_top1_5000")
        struc_feat = np.load("saved/h-index-top-1.npy")
        self.data.x = torch.cat([self.data.x, torch.from_numpy(struc_feat)], dim=1)

@register_dataset("h-index-rand20intop200-attr")
class HIndexTop200Attr(EdgelistWithAttr):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_rand20intop200_5000")

@register_dataset("h-index-rand20intop200-attr-cat-struc")
class HIndexTop200Attr(EdgelistWithAttr):
    def __init__(self):
        super().__init__("cogdl/data/hindex/", "aminer_hindex_rand20intop200_5000")
        struc_feat = np.load("saved/h-index-rand20intop200.npy")
        self.data.x = torch.cat([self.data.x, torch.from_numpy(struc_feat)], dim=1)
