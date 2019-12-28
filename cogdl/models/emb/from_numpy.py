import numpy as np
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
import random
from .. import BaseModel, register_model

from .prone import ProNE


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
        return cls(
            args.hidden_size,
            args.emb_path,
            args,
        )

    def __init__(self, hidden_size, emb_path, args):
        super(FromNumpy, self).__init__()
        self.hidden_size = hidden_size
        self.emb = np.load(emb_path)

        # HACK
        # args.hidden_size = 32
        # self.prone = ProNE.build_model_from_args(args)
        # args.hidden_size = 64

    def train(self, G):
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        assert G.number_of_nodes() == embeddings.shape[0]
        # embeddings = embeddings.T
        embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)
        # embeddings = embeddings / (embeddings.std() + 1e-8)
        # embeddings = embeddings.T

        a = np.arange(embeddings.shape[0])
        np.random.shuffle(a)
        embeddings = embeddings[a]
        # return np.random.normal(0, 1, embeddings.shape)
        return embeddings

@register_model("from_numpy_cat_prone")
class FromNumpyCatProne(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--emb-path', type=str,
                            help='Load embeddings from npy file')
        parser.add_argument("--step", type=int, default=5,
                            help=" Number of items in the chebyshev expansion")
        parser.add_argument("--mu", type=float, default=0.2)
        parser.add_argument("--theta", type=float, default=0.5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(
            args.hidden_size,
            args.emb_path,
            args,
        )

    def __init__(self, hidden_size, emb_path, args):
        super(FromNumpyCatProne, self).__init__()
        self.hidden_size = hidden_size
        self.emb = np.load(emb_path)

        # HACK
        args.hidden_size //= 2
        self.prone = ProNE.build_model_from_args(args)
        args.hidden_size *= 2

    def train(self, G):
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        embeddings = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        assert G.number_of_nodes() == embeddings.shape[0]
        embeddings = (embeddings - embeddings.mean(axis=0)) / (embeddings.std(axis=0) + 1e-8)

        prone_embeddings = self.prone.train(G)

        return np.concatenate([embeddings, prone_embeddings], axis=1)

