import numpy as np

from .. import BaseModel, register_model
from ._graphwave import graphwave_alg
from .prone import ProNE


@register_model("graphwave")
class GraphWave(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--scale", type=float, default=1e5)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.scale, args)

    def __init__(self, dimension, scale, args):
        super(GraphWave, self).__init__()
        self.dimension = dimension
        self.scale = scale
        self.whitening = args.task == "unsupervised_node_classification"

    def train(self, G):
        chi, heat_print, taus = graphwave_alg(
            G, np.linspace(0, self.scale, self.dimension // 4)
        )
        # if self.whitening:
        #     chi = (chi - chi.mean(axis=0)) / (chi.std(axis=0) + 1e-8)
        return chi


@register_model("graphwave_cat_prone")
class GraphwaveCatProne(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--scale", type=float, default=1e5)
        ProNE.add_args(parser)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args.hidden_size, args.scale, args)

    def __init__(self, dimension, scale, args):
        super(GraphwaveCatProne, self).__init__()
        self.dimension = dimension // 2
        self.scale = scale
        self.whitening = args.task == "unsupervised_node_classification"

        # HACK
        args.hidden_size //= 2
        self.prone = ProNE.build_model_from_args(args)
        args.hidden_size *= 2

    def train(self, G):
        chi, heat_print, taus = graphwave_alg(
            G, np.linspace(0, self.scale, self.dimension // 4)
        )
        # if self.whitening:
        #     chi = (chi - chi.mean(axis=0)) / (chi.std(axis=0) + 1e-8)
        prone_embeddings = self.prone.train(G)

        return np.concatenate([chi, prone_embeddings], axis=1)
