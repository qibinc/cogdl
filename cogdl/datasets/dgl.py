import torch

import dgl
from cogdl.data import Data, Dataset, download_url
from dgl.data import AmazonCoBuy, Coauthor

from . import register_dataset


@register_dataset("dgl")
class DGLDataset(Dataset):

    def __init__(self):
        graphs = []
        for name in ["cs", "physics"]:
            g = Coauthor(name)[0]
            graphs.append(g)
        for name in ["computers", "photo"]:
            g = AmazonCoBuy(name)[0]
            graphs.append(g)
        # more graphs are comming ...

        self.graph = dgl.batch(graphs, node_attrs=None, edge_attrs=None)
        self.graph.remove_nodes((self.graph.in_degrees() == 0).nonzero().squeeze())
        print(self.graph.number_of_nodes())
        self.data = Data(edge_index=torch.stack(self.graph.edges()))

    def __getitem__(self, idx):
        return self.data

    def __len__(self):
        return 1
