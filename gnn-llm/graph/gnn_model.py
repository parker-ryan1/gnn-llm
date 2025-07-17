import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from typing import List, Tuple, Dict
import logging

class DistanceMinimizingGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 3):
        super(DistanceMinimizingGNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer for node embeddings
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)
        
        # Distance prediction head
        self.distance_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        x, edge_index = data.x, data.edge_index
        
        # Apply graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        # Final node embeddings
        node_embeddings = self.output_layer(x)
        
        # Predict distances for all node pairs
        distances = self._predict_pairwise_distances(node_embeddings)
        
        return node_embeddings, distances
    
    def _predict_pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Predict distances between all pairs of nodes"""
        num_nodes = embeddings.size(0)
        distances = torch.zeros(num_nodes, num_nodes, device=embeddings.device)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Concatenate embeddings for distance prediction
                pair_embedding = torch.cat([embeddings[i], embeddings[j]], dim=0)
                distance = self.distance_predictor(pair_embedding)
                distances[i, j] = distance
                distances[j, i] = distance  # Symmetric
        
        return distances

class GNNTrainer:
    def __init__(self, model: DistanceMinimizingGNN, learning_rate: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.logger = logging.getLogger(__name__)
        
    def distance_minimization_loss(self, predicted_distances: torch.Tensor, 
                                 actual_distances: torch.Tensor, 
                                 edge_index: torch.Tensor) -> torch.Tensor:
        """
        Loss function that rewards the model for finding ways to shrink distances
        between connected nodes while maintaining separation for distant nodes
        """
        # Extract distances for connected nodes (edges)
        edge_distances = predicted_distances[edge_index[0], edge_index[1]]
        actual_edge_distances = actual_distances[edge_index[0], edge_index[1]]
        
        # Reward for minimizing distances between connected nodes
        connection_loss = F.mse_loss(edge_distances, torch.zeros_like(edge_distances))
        
        # Penalty for making all distances too small (maintain some separation)
        separation_loss = torch.mean(torch.relu(0.1 - predicted_distances))
        
        # Reconstruction loss (maintain some similarity to original distances)
        reconstruction_loss = F.mse_loss(predicted_distances, actual_distances)
        
        # Combined loss with weights
        total_loss = connection_loss + 0.1 * separation_loss + 0.5 * reconstruction_loss
        
        return total_loss
    
    def calculate_actual_distances(self, data: Data) -> torch.Tensor:
        """Calculate actual distances in the embedding space"""
        embeddings = data.x
        num_nodes = embeddings.size(0)
        distances = torch.zeros(num_nodes, num_nodes, device=embeddings.device)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # Euclidean distance in embedding space
                distance = torch.norm(embeddings[i] - embeddings[j])
                distances[i, j] = distance
                distances[j, i] = distance
        
        # Normalize distances to [0, 1]
        max_distance = torch.max(distances)
        if max_distance > 0:
            distances = distances / max_distance
        
        return distances
    
    def train_epoch(self, data: Data) -> float:
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        node_embeddings, predicted_distances = self.model(data)
        
        # Calculate actual distances
        actual_distances = self.calculate_actual_distances(data)
        
        # Calculate loss
        loss = self.distance_minimization_loss(
            predicted_distances, actual_distances, data.edge_index
        )
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, data: Data, num_epochs: int = 100) -> List[float]:
        """Train the model"""
        losses = []
        
        for epoch in range(num_epochs):
            loss = self.train_epoch(data)
            losses.append(loss)
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss:.4f}")
        
        return losses
    
    def get_optimized_embeddings(self, data: Data) -> np.ndarray:
        """Get optimized node embeddings after training"""
        self.model.eval()
        with torch.no_grad():
            node_embeddings, _ = self.model(data)
            return node_embeddings.cpu().numpy()