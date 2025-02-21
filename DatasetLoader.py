import torch
import pandas as pd

def prepdata(dataset_path = "./Datasets/rec-libimseti-dir.edges"):
        df = pd.read_csv(dataset_path, sep='\t', header=None, names=['Node1', 'Node2', 'Weight'])

        # Weights
        edge_weight = torch.tensor(df['Weight'].values)

        # Edge Index
        edge_index = torch.tensor(df[["Node1", "Node2"]].values.T) - 1

        # Node amount
        node_amount = len(edge_index.unique())

        # edge_index (2, E)
        # Weights (E)
        return edge_index, edge_weight, node_amount

def pickRandomEdge(edge_index, edge_weight, node_amount, device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        num_edges = edge_weight.shape[0]
        rand_index = torch.randint(0, node_amount, (1,)).item()
        features = torch.zeros(node_amount)
        features[rand_index] = 1
        features = features.unsqueeze(1) 

        mask = edge_index[1, :] != rand_index

        # Use the mask to prune the edges and corresponding weights.
        pruned_edge_index = edge_index[:, mask]
        pruned_edge_weight = edge_weight[mask].float()

        return pruned_edge_index.to(device), pruned_edge_weight.to(device), features.to(device), rand_index



if __name__ == "__main__":
        prepped = prepdata()
        edge_index = prepped[0]
        edge_weight = prepped[1]
        print("NODE AMOUNT: {}".format(prepped[2]))
        print("Edge Index shape: {}".format(prepped[0].shape))
        print("Edge Weights shape: {}".format(prepped[1].shape))

        pruned_data = pickRandomEdge(edge_index, edge_weight, prepped[2])

        print("PRUNED")
        print(pruned_data[0].shape)
        print(pruned_data[1].shape)
        print(torch.argmax(pruned_data[2]))
        print(pruned_data[3])


    