import os.path as osp

import numpy as np
import torch

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, Reddit, TUDataset

from . import register_dataset


@register_dataset("cora")
class CoraDataset(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(CoraDataset, self).__init__(path, dataset, T.TargetIndegree())

@register_dataset("cora_struc")
class CoraStrucDataset(Planetoid):
    def __init__(self):
        dataset = "Cora"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(CoraStrucDataset, self).__init__(path, dataset, T.TargetIndegree())
        struc_feat = np.load("saved/cora.npy")
        self.data.x = torch.cat([self.data.x, torch.from_numpy(struc_feat)], dim=1)

@register_dataset("citeseer")
class CiteSeerDataset(Planetoid):
    def __init__(self):
        dataset = "CiteSeer"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(CiteSeerDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("pubmed")
class PubMedDataset(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(PubMedDataset, self).__init__(path, dataset, T.TargetIndegree())


@register_dataset("pubmed_struc")
class PubMedStrucDataset(Planetoid):
    def __init__(self):
        dataset = "PubMed"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(PubMedStrucDataset, self).__init__(path, dataset, T.TargetIndegree())
        struc_feat = np.load("saved/pubmed.npy")
        self.data.x = torch.cat([self.data.x, torch.from_numpy(struc_feat)], dim=1)


@register_dataset("reddit")
class RedditDataset(Reddit):
    def __init__(self):
        dataset = "Reddit"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(RedditDataset, self).__init__(path, T.TargetIndegree())

@register_dataset("collab")
class CollabDataset(TUDataset):
    def __init__(self):
        dataset = "COLLAB"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(CollabDataset, self).__init__(path, dataset)

@register_dataset("imdb-binary")
class IMDBBinaryDataset(TUDataset):
    def __init__(self):
        dataset = "IMDB-BINARY"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(IMDBBinaryDataset, self).__init__(path, dataset)

@register_dataset("imdb-multi")
class IMDBMultiDataset(TUDataset):
    def __init__(self):
        dataset = "IMDB-MULTI"
        path = osp.join(osp.dirname(osp.realpath(__file__)), "../..", "data", dataset)
        super(IMDBMultiDataset, self).__init__(path, dataset)