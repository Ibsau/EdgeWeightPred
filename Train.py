import torch
from DatasetLoader import prepdata, pickRandomEdge
from Model import GCNModel

# Test forward pass
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = GCNModel(input_dim=1, hidden_dim=64, output_dim=10).to(device)

data = prepdata()
pruned_edge_index, pruned_edge_weight, features, rand_index = pickRandomEdge(*data)

output = model(features, pruned_edge_index, pruned_edge_weight)
print(output)
print(output.shape)

