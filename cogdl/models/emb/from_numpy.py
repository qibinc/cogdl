import random

import networkx as nx
import numpy as np
from gensim.models import KeyedVectors, Word2Vec

from .. import BaseModel, register_model
from .prone import ProNE

@register_model("zero")
class Zero(BaseModel):

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size)

    def __init__(self, hidden_size):
        super(Zero, self).__init__()
        self.hidden_size = hidden_size

    def train(self, G):
        return np.zeros((G.number_of_nodes(), self.hidden_size))

@register_model("from_numpy")
class FromNumpy(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--emb-path', type=str,
                            help='Load embeddings from npy file')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.emb_path, args)

    def __init__(self, hidden_size, emb_path, args):
        super(FromNumpy, self).__init__()
        self.hidden_size = hidden_size
        self.emb = np.load(emb_path)
        self.whitening = args.task == "unsupervised_node_classification"

        # HACK
        # args.hidden_size = 32
        # self.prone = ProNE.build_model_from_args(args)
        # args.hidden_size = 64

    def train(self, G):
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        assert G.number_of_nodes() == embeddings.shape[0]
        # embeddings = embeddings.T
        # if self.whitening:
        #     embeddings = (embeddings - embeddings.mean(axis=0)) / (
        #         embeddings.std(axis=0) + 1e-8
        #     )
        # embeddings = embeddings / (embeddings.std() + 1e-8)
        # embeddings = embeddings.T

        # a = np.arange(embeddings.shape[0])
        # np.random.shuffle(a)
        # embeddings = embeddings[a]
        # return np.random.normal(0, 1, embeddings.shape)
        return embeddings

@register_model("from_numpy_graph")
class FromNumpyGraph(FromNumpy):

    def train(self, G):
        assert G is None
        return self.emb

@register_model("from_numpy_cat_prone")
class FromNumpyCatProne(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--emb-path', type=str,
                            help='Load embeddings from npy file')
        ProNE.add_args(parser)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.emb_path, args)

    def __init__(self, hidden_size, emb_path, args):
        super(FromNumpyCatProne, self).__init__()
        self.hidden_size = hidden_size // 2
        self.emb = np.load(emb_path)
        self.whitening = args.task == "unsupervised_node_classification"

        # HACK
        args.hidden_size //= 2
        self.prone = ProNE.build_model_from_args(args)
        args.hidden_size *= 2

    def train(self, G):
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        assert G.number_of_nodes() == embeddings.shape[0]
        # if self.whitening:
        #     embeddings = (embeddings - embeddings.mean(axis=0)) / (
        #         embeddings.std(axis=0) + 1e-8
        #     )

        prone_embeddings = self.prone.train(G)

        return np.concatenate([embeddings, prone_embeddings], axis=1)


@register_model("from_numpy_align")
class FromNumpyAlign(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--emb-path-1', type=str,
                            help='Load embeddings from npy file')
        parser.add_argument('--emb-path-2', type=str,
                            help='Load embeddings from npy file')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.emb_path_1, args.emb_path_2)

    def __init__(self, hidden_size, emb_path_1, emb_path_2):
        super(FromNumpyAlign, self).__init__()
        self.hidden_size = hidden_size
        self.emb_1 = np.load(emb_path_1)
        self.emb_2 = np.load(emb_path_2)
        self.t1, self.t2 = False, False

    def train(self, G):
        if G.number_of_nodes() == self.emb_1.shape[0] and not self.t1:
            emb = self.emb_1
            self.t1 = True
        elif G.number_of_nodes() == self.emb_2.shape[0] and not self.t2:
            emb = self.emb_2
            self.t2 = True
        else:
            raise NotImplementedError

        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([emb[id2node[i]] for i in range(len(id2node))])

        return embeddings
