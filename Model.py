import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class GCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=10):
        super(GCNModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)  # Output single value
    
    def forward(self, x, edge_index, edge_weight):
        x_out = self.gcn1(x, edge_index, edge_weight)
        x_out = self.gcn2(x_out, edge_index, edge_weight)
        x_out = self.fc(x_out)
        # Use sigmoid activation to smoothly constrain the output to [0, 10]
        x_out = torch.sigmoid(x_out) * self.output_dim
        return x_out
