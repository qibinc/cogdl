import numpy as np
import networkx as nx
from gensim.models import Word2Vec, KeyedVectors
import random
from .. import BaseModel, register_model

from .prone import ProNE


@register_model("from_numpy")
class DeepWalk(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--walk-length', type=int, default=80,
                            help='Length of walk per source. Default is 80.')
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
        super(DeepWalk, self).__init__()
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
        # return np.random.normal(0, 1, embeddings.shape)
        # return embeddings / 100
        return embeddings

        # HACK
        # prone_embeddings = self.prone.train(G)
        # print(abs(embeddings).mean(), abs(prone_embeddings).mean())
        # return np.concatenate([embeddings / 100, prone_embeddings], axis=1)

