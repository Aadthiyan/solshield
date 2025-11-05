#!/usr/bin/env python3
"""
Joint Syntax-Semantic Graph Learning for Smart Contract Vulnerability Detection

This module implements a hybrid graph learning approach that combines:
1. Detailed syntax trees (AST subtrees) as graph nodes
2. Semantic relationships (control flow, data flow) as graph edges
3. Hierarchical attention mechanisms for syntax-semantic interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import ast
import re

@dataclass
class GraphNode:
    """Represents a node in the joint syntax-semantic graph"""
    node_id: str
    node_type: str  # 'function', 'variable', 'statement', 'expression'
    syntax_subtree: ast.AST  # AST subtree for this node
    semantic_features: Dict[str, Any]  # Control flow, data flow features
    position: Tuple[int, int]  # Line numbers
    importance_score: float = 0.0

@dataclass
class GraphEdge:
    """Represents an edge in the joint syntax-semantic graph"""
    source_id: str
    target_id: str
    edge_type: str  # 'control_flow', 'data_flow', 'call_flow', 'dependency'
    weight: float = 1.0
    semantic_context: Dict[str, Any] = None

class SyntaxSemanticGraphBuilder:
    """Builds joint syntax-semantic graphs from Solidity code"""
    
    def __init__(self):
        self.node_counter = 0
        self.function_patterns = [
            r'function\s+\w+\s*\([^)]*\)\s*(?:public|private|internal|external)',
            r'modifier\s+\w+\s*\([^)]*\)',
            r'constructor\s*\([^)]*\)'
        ]
        self.vulnerability_patterns = [
            r'\.call\s*\(',
            r'\.send\s*\(',
            r'\.transfer\s*\(',
            r'selfdestruct\s*\(',
            r'require\s*\(',
            r'assert\s*\('
        ]
    
    def build_graph(self, contract_code: str) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """Build joint syntax-semantic graph from contract code"""
        try:
            # Parse AST
            tree = ast.parse(contract_code)
            
            # Extract nodes and edges
            nodes = self._extract_nodes(tree, contract_code)
            edges = self._extract_edges(nodes, contract_code)
            
            # Add semantic relationships
            edges.extend(self._extract_semantic_relationships(nodes, contract_code))
            
            return nodes, edges
            
        except Exception as e:
            print(f"Error building graph: {e}")
            return [], []
    
    def _extract_nodes(self, tree: ast.AST, code: str) -> List[GraphNode]:
        """Extract nodes from AST with syntax subtrees"""
        nodes = []
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign, ast.Call)):
                node_id = f"node_{self.node_counter}"
                self.node_counter += 1
                
                # Extract syntax subtree
                syntax_subtree = node
                
                # Extract semantic features
                semantic_features = self._extract_semantic_features(node, code)
                
                # Get position
                position = (getattr(node, 'lineno', 0), getattr(node, 'end_lineno', 0))
                
                # Calculate importance score
                importance_score = self._calculate_importance(node, code)
                
                graph_node = GraphNode(
                    node_id=node_id,
                    node_type=type(node).__name__,
                    syntax_subtree=syntax_subtree,
                    semantic_features=semantic_features,
                    position=position,
                    importance_score=importance_score
                )
                
                nodes.append(graph_node)
        
        return nodes
    
    def _extract_edges(self, nodes: List[GraphNode], code: str) -> List[GraphEdge]:
        """Extract control flow and data flow edges"""
        edges = []
        
        # Control flow edges
        for i, node in enumerate(nodes):
            for j, other_node in enumerate(nodes):
                if i != j:
                    # Check for control flow relationships
                    if self._has_control_flow(node, other_node, code):
                        edge = GraphEdge(
                            source_id=node.node_id,
                            target_id=other_node.node_id,
                            edge_type='control_flow',
                            weight=self._calculate_control_flow_weight(node, other_node)
                        )
                        edges.append(edge)
                    
                    # Check for data flow relationships
                    if self._has_data_flow(node, other_node, code):
                        edge = GraphEdge(
                            source_id=node.node_id,
                            target_id=other_node.node_id,
                            edge_type='data_flow',
                            weight=self._calculate_data_flow_weight(node, other_node)
                        )
                        edges.append(edge)
        
        return edges
    
    def _extract_semantic_relationships(self, nodes: List[GraphNode], code: str) -> List[GraphEdge]:
        """Extract semantic relationships between nodes"""
        edges = []
        
        # Function call relationships
        for node in nodes:
            if node.node_type == 'Call':
                for other_node in nodes:
                    if other_node.node_type == 'FunctionDef':
                        if self._is_function_call(node, other_node, code):
                            edge = GraphEdge(
                                source_id=node.node_id,
                                target_id=other_node.node_id,
                                edge_type='call_flow',
                                weight=1.0
                            )
                            edges.append(edge)
        
        # Variable dependency relationships
        for node in nodes:
            if node.node_type == 'Assign':
                for other_node in nodes:
                    if self._has_variable_dependency(node, other_node, code):
                        edge = GraphEdge(
                            source_id=node.node_id,
                            target_id=other_node.node_id,
                            edge_type='dependency',
                            weight=0.8
                        )
                        edges.append(edge)
        
        return edges
    
    def _extract_semantic_features(self, node: ast.AST, code: str) -> Dict[str, Any]:
        """Extract semantic features from AST node"""
        features = {
            'has_external_call': False,
            'has_state_modification': False,
            'has_loop': False,
            'has_condition': False,
            'security_patterns': [],
            'vulnerability_indicators': []
        }
        
        # Check for external calls
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'attr'):
                    if child.func.attr in ['call', 'send', 'transfer']:
                        features['has_external_call'] = True
        
        # Check for state modifications
        if isinstance(node, ast.Assign):
            features['has_state_modification'] = True
        
        # Check for loops
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                features['has_loop'] = True
        
        # Check for conditions
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                features['has_condition'] = True
        
        # Check for security patterns
        node_code = ast.get_source_segment(code, node) or ""
        for pattern in self.vulnerability_patterns:
            if re.search(pattern, node_code):
                features['vulnerability_indicators'].append(pattern)
        
        return features
    
    def _calculate_importance(self, node: ast.AST, code: str) -> float:
        """Calculate importance score for a node"""
        score = 0.0
        
        # Function definitions are more important
        if isinstance(node, ast.FunctionDef):
            score += 0.5
        
        # External calls are important
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if hasattr(child.func, 'attr'):
                    if child.func.attr in ['call', 'send', 'transfer']:
                        score += 0.3
        
        # State modifications are important
        if isinstance(node, ast.Assign):
            score += 0.2
        
        return min(score, 1.0)
    
    def _has_control_flow(self, node1: GraphNode, node2: GraphNode, code: str) -> bool:
        """Check if there's a control flow relationship between nodes"""
        # Simple heuristic: if node2 comes after node1 in the same function
        if (node1.position[1] < node2.position[0] and 
            abs(node1.position[1] - node2.position[0]) < 10):
            return True
        return False
    
    def _has_data_flow(self, node1: GraphNode, node2: GraphNode, code: str) -> bool:
        """Check if there's a data flow relationship between nodes"""
        # Check if node2 uses variables defined in node1
        if node1.node_type == 'Assign' and node2.node_type == 'Call':
            return True
        return False
    
    def _is_function_call(self, call_node: GraphNode, func_node: GraphNode, code: str) -> bool:
        """Check if call_node is calling func_node"""
        # Simplified check - in real implementation, would parse function names
        return True
    
    def _has_variable_dependency(self, assign_node: GraphNode, other_node: GraphNode, code: str) -> bool:
        """Check if other_node depends on variables from assign_node"""
        # Simplified check - in real implementation, would track variable usage
        return True
    
    def _calculate_control_flow_weight(self, node1: GraphNode, node2: GraphNode) -> float:
        """Calculate weight for control flow edge"""
        return 0.8
    
    def _calculate_data_flow_weight(self, node1: GraphNode, node2: GraphNode) -> float:
        """Calculate weight for data flow edge"""
        return 0.6

class HierarchicalSyntaxSemanticGNN(nn.Module):
    """Hierarchical GNN for joint syntax-semantic learning"""
    
    def __init__(self, 
                 input_dim: int = 128,
                 hidden_dim: int = 256,
                 output_dim: int = 64,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Syntax feature extractor
        self.syntax_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Semantic feature extractor
        self.semantic_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Joint attention mechanism
        self.joint_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Hierarchical GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.gnn_layers.append(
                    TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
                )
            else:
                self.gnn_layers.append(
                    TransformerConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
                )
        
        # Syntax-semantic fusion layers
        self.fusion_layers = nn.ModuleList()
        for i in range(num_layers):
            self.fusion_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Vulnerability detection head
        self.vulnerability_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: torch.Tensor, batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the hierarchical GNN"""
        
        # Extract syntax and semantic features
        syntax_features = self.syntax_encoder(x)
        semantic_features = self.semantic_encoder(x)
        
        # Joint attention between syntax and semantic features
        joint_features, attention_weights = self.joint_attention(
            syntax_features.unsqueeze(0),
            semantic_features.unsqueeze(0),
            semantic_features.unsqueeze(0)
        )
        joint_features = joint_features.squeeze(0)
        
        # Hierarchical GNN processing
        node_features = joint_features
        layer_outputs = []
        
        for i, (gnn_layer, fusion_layer) in enumerate(zip(self.gnn_layers, self.fusion_layers)):
            # GNN layer
            gnn_output = gnn_layer(node_features, edge_index, edge_attr)
            
            # Fusion with previous layer
            if i > 0:
                fused_features = fusion_layer(torch.cat([gnn_output, node_features], dim=-1))
            else:
                fused_features = gnn_output
            
            node_features = fused_features
            layer_outputs.append(node_features)
        
        # Global pooling
        graph_embedding = global_mean_pool(node_features, batch)
        
        # Classification
        node_classification = self.classifier(node_features)
        vulnerability_prediction = self.vulnerability_head(node_features)
        
        return {
            'node_embeddings': node_features,
            'graph_embedding': graph_embedding,
            'node_classification': node_classification,
            'vulnerability_prediction': vulnerability_prediction,
            'attention_weights': attention_weights,
            'layer_outputs': layer_outputs
        }

class JointGraphDataProcessor:
    """Processes joint syntax-semantic graphs for training"""
    
    def __init__(self, max_nodes: int = 1000, max_edges: int = 2000):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.graph_builder = SyntaxSemanticGraphBuilder()
    
    def process_contract(self, contract_code: str, label: int) -> Data:
        """Process a contract into a joint graph"""
        nodes, edges = self.graph_builder.build_graph(contract_code)
        
        if not nodes:
            # Return empty graph if parsing fails
            return Data(x=torch.zeros(1, 128), edge_index=torch.zeros(2, 0, dtype=torch.long))
        
        # Convert to PyTorch Geometric format
        node_features = self._extract_node_features(nodes)
        edge_index, edge_attr = self._extract_edge_features(edges, nodes)
        
        # Create graph data
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor([label], dtype=torch.float)
        )
        
        return graph_data
    
    def _extract_node_features(self, nodes: List[GraphNode]) -> torch.Tensor:
        """Extract features from graph nodes"""
        features = []
        
        for node in nodes:
            # Basic node features
            node_feature = [
                float(node.importance_score),
                float(len(node.semantic_features.get('vulnerability_indicators', []))),
                float(node.semantic_features.get('has_external_call', False)),
                float(node.semantic_features.get('has_state_modification', False)),
                float(node.semantic_features.get('has_loop', False)),
                float(node.semantic_features.get('has_condition', False)),
            ]
            
            # Add position features
            node_feature.extend([
                float(node.position[0]) / 1000.0,  # Normalized line number
                float(node.position[1] - node.position[0]) / 100.0  # Normalized span
            ])
            
            # Pad to fixed size
            while len(node_feature) < 128:
                node_feature.append(0.0)
            
            features.append(node_feature[:128])
        
        return torch.tensor(features, dtype=torch.float)
    
    def _extract_edge_features(self, edges: List[GraphEdge], nodes: List[GraphNode]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract edge features and create edge index"""
        if not edges:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4)
        
        # Create node ID to index mapping
        node_id_to_idx = {node.node_id: i for i, node in enumerate(nodes)}
        
        edge_indices = []
        edge_attrs = []
        
        for edge in edges:
            if edge.source_id in node_id_to_idx and edge.target_id in node_id_to_idx:
                source_idx = node_id_to_idx[edge.source_id]
                target_idx = node_id_to_idx[edge.target_id]
                
                edge_indices.append([source_idx, target_idx])
                
                # Edge features
                edge_attr = [
                    float(edge.weight),
                    float(edge.edge_type == 'control_flow'),
                    float(edge.edge_type == 'data_flow'),
                    float(edge.edge_type == 'call_flow')
                ]
                edge_attrs.append(edge_attr)
        
        if not edge_indices:
            return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0, 4)
        
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        
        return edge_index, edge_attr

# Example usage and testing
if __name__ == "__main__":
    # Test the joint graph learning system
    sample_contract = """
    pragma solidity ^0.8.0;
    
    contract TestContract {
        mapping(address => uint256) public balances;
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            balances[msg.sender] -= amount;
        }
    }
    """
    
    # Build joint graph
    builder = SyntaxSemanticGraphBuilder()
    nodes, edges = builder.build_graph(sample_contract)
    
    print(f"Built graph with {len(nodes)} nodes and {len(edges)} edges")
    
    # Process for model input
    processor = JointGraphDataProcessor()
    graph_data = processor.process_contract(sample_contract, label=1)
    
    print(f"Processed graph: {graph_data.x.shape}, {graph_data.edge_index.shape}")
    
    # Test model
    model = HierarchicalSyntaxSemanticGNN()
    output = model(graph_data.x, graph_data.edge_index, 
                   graph_data.edge_attr if hasattr(graph_data, 'edge_attr') else torch.zeros(graph_data.edge_index.shape[1], 4),
                   torch.zeros(graph_data.x.shape[0], dtype=torch.long))
    
    print(f"Model output keys: {output.keys()}")
    print(f"Vulnerability prediction shape: {output['vulnerability_prediction'].shape}")
