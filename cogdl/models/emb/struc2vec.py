import numpy as np

from cogdl.datasets import build_dataset

from .. import BaseModel, register_model
from .prone import ProNE


@register_model("struc2vec")
class Struc2vec(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--emb-path', type=str,
                            help='Load self.emb from npy file')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(Struc2vec, self).__init__()

        node2id = build_dataset(args).node2id
        with open(args.emb_path) as f:
            num_nodes, dim = list(map(int, f.readline().strip().split()))
            assert len(node2id) == num_nodes and dim == args.hidden_size, "Dataset and emb dimension doesn't match"
            self.emb = np.zeros((num_nodes, dim))
            for line in f:
                line = line.strip().split()
                x = node2id[int(line[0])]
                embedding = np.array(list(map(float, line[1:])))
                self.emb[x] = embedding
        # if args.task == "unsupervised_node_classification":
        #     self.emb = (self.emb - self.emb.mean(axis=0)) / (self.emb.std(axis=0) + 1e-8)

    def train(self, G):
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        self.emb = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        return self.emb


@register_model("struc2vec_cat_prone")
class Struc2vecCatProne(Struc2vec):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        Struc2vec.add_args(parser)
        ProNE.add_args(parser)
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        args.hidden_size //= 2
        super(Struc2vecCatProne, self).__init__(args)
        self.prone = ProNE.build_model_from_args(args)
        args.hidden_size *= 2

    def train(self, G):
        id2node = dict([(vid, node) for vid, node in enumerate(G.nodes())])
        self.emb = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        prone_embeddings = self.prone.train(G)
        return np.concatenate([self.emb, prone_embeddings], axis=1)

@register_model("struc2vec_align")
class Struc2vecAlign(BaseModel):
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--emb-path-1', type=str,
                            help='Load self.emb from npy file')
        parser.add_argument('--emb-path-2', type=str,
                            help='Load self.emb from npy file')
        # fmt: on

    @classmethod
    def build_model_from_args(cls, args):
        return cls(args)

    def __init__(self, args):
        super(Struc2vecAlign, self).__init__()
        dataset = build_dataset(args)
        self.emb_1 = self._load_emb(args.emb_path_1, dataset.node2id_1, args.hidden_size)
        self.emb_2 = self._load_emb(args.emb_path_2, dataset.node2id_2, args.hidden_size)
        self.t1, self.t2 = False, False

    def _load_emb(self, emb_path, node2id, hidden_size):
        with open(emb_path) as f:
            num_nodes, dim = list(map(int, f.readline().strip().split()))
            assert len(node2id) == num_nodes and dim == hidden_size, "Dataset and emb dimension doesn't match"
            emb = np.zeros((num_nodes, dim))
            for line in f:
                line = line.strip().split()
                x = node2id[int(line[0])]
                embedding = np.array(list(map(float, line[1:])))
                emb[x] = embedding
        return emb

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
        self.emb = np.asarray([self.emb[id2node[i]] for i in range(len(id2node))])
        return self.emb