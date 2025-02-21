import torch
import torch.nn as nn
import torch.optim as optim
from DatasetLoader import prepdata, pickRandomEdge
from Model import GCNModel

def main():
    # Set device: CUDA if available, otherwise CPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Load the dataset: edge_index, edge_weight, and number of nodes.
    original_edge_index, original_edge_weight, node_amount = prepdata()
    
    # Initialize the model, optimizer, and loss function.
    model = GCNModel(input_dim=1, hidden_dim=64, output_dim=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    n_epochs = 1000
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Sample a random node and prune edges directed to that node.
        pruned_edge_index, pruned_edge_weight, features, rand_index = pickRandomEdge(
            original_edge_index, original_edge_weight, node_amount, device=device
        )
        
        # Forward pass: compute model output for all nodes.
        output = model(features, pruned_edge_index, pruned_edge_weight)
        # Get the predicted value for the randomly selected node.
        predicted = output[rand_index]
        
        # Compute the target: average of original edge weights directed to that node.
        target_mask = (original_edge_index[1, :] == rand_index)
        if target_mask.sum() > 0:
            target_edge_weights = original_edge_weight[target_mask].float()
            target = target_edge_weights.mean().to(device)
        else:
            target = torch.tensor(0.0, device=device)
        
        # Compute loss between predicted value and target value.
        loss = loss_fn(predicted, target)
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Loss: {loss.item():.4f} | Target: {target.item():.4f} | Predicted: {predicted.item():.4f}")
    
    # Final evaluation on a new sample.
    model.eval()
    pruned_edge_index, pruned_edge_weight, features, rand_index = pickRandomEdge(
        original_edge_index, original_edge_weight, node_amount, device=device
    )
    output = model(features, pruned_edge_index, pruned_edge_weight)
    predicted = output[rand_index]
    print("\nFinal prediction for node {}: {}".format(rand_index, predicted.item()))

if __name__ == "__main__":
    main()
