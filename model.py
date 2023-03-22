from torch_geometric.nn import GCNConv, SAGEConv
import torch.nn.functional as F
import torch

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels,features_in, features_out):
        super().__init__()
        self.conv1 = GCNConv(features_in, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, features_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, hidden_channels,features_in, features_out):
        super().__init__()
        self.conv1 = SAGEConv(features_in, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, features_out)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x