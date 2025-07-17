import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import torch
from torch_geometric.data import Data
import logging

class GraphBuilder:
    def __init__(self, similarity_threshold: float = 0.7, max_edges_per_node: int = 10):
        self.similarity_threshold = similarity_threshold
        self.max_edges_per_node = max_edges_per_node
        self.logger = logging.getLogger(__name__)
    
    def build_graph_from_embeddings(self, embeddings: Dict[str, np.ndarray]) -> nx.Graph:
        """Build NetworkX graph from embeddings using cosine similarity"""
        G = nx.Graph()
        node_ids = list(embeddings.keys())
        embedding_matrix = np.array(list(embeddings.values()))
        
        # Add nodes
        for node_id in node_ids:
            G.add_node(node_id, embedding=embeddings[node_id])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embedding_matrix)
        
        # Add edges based on similarity
        for i, node_i in enumerate(node_ids):
            # Get top similar nodes (excluding self)
            similarities = similarity_matrix[i]
            similar_indices = np.argsort(similarities)[::-1][1:self.max_edges_per_node + 1]
            
            for j in similar_indices:
                similarity_score = similarities[j]
                if similarity_score >= self.similarity_threshold:
                    node_j = node_ids[j]
                    if not G.has_edge(node_i, node_j):
                        G.add_edge(node_i, node_j, weight=similarity_score)
        
        self.logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        return G
    
    def networkx_to_pytorch_geometric(self, G: nx.Graph) -> Data:
        """Convert NetworkX graph to PyTorch Geometric format"""
        # Create node mapping
        node_mapping = {node: i for i, node in enumerate(G.nodes())}
        
        # Extract node features (embeddings)
        node_features = []
        for node in G.nodes():
            embedding = G.nodes[node]['embedding']
            node_features.append(embedding)
        
        x = torch.tensor(np.array(node_features), dtype=torch.float)
        
        # Extract edges
        edge_list = []
        edge_weights = []
        
        for edge in G.edges(data=True):
            source, target, data = edge
            edge_list.append([node_mapping[source], node_mapping[target]])
            edge_list.append([node_mapping[target], node_mapping[source]])  # Undirected
            
            weight = data.get('weight', 1.0)
            edge_weights.extend([weight, weight])
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    def calculate_graph_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """Calculate various graph metrics"""
        metrics = {}
        
        if G.number_of_nodes() > 0:
            metrics['num_nodes'] = G.number_of_nodes()
            metrics['num_edges'] = G.number_of_edges()
            metrics['density'] = nx.density(G)
            
            if nx.is_connected(G):
                metrics['average_path_length'] = nx.average_shortest_path_length(G)
                metrics['diameter'] = nx.diameter(G)
            else:
                # For disconnected graphs, calculate for largest component
                largest_cc = max(nx.connected_components(G), key=len)
                subgraph = G.subgraph(largest_cc)
                metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
                metrics['diameter'] = nx.diameter(subgraph)
                metrics['num_components'] = nx.number_connected_components(G)
            
            metrics['clustering_coefficient'] = nx.average_clustering(G)
        
        return metrics