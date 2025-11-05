#!/usr/bin/env python3
"""
Graph Neural Network Model for Smart Contract Vulnerability Detection

This module implements a GNN-based model for detecting vulnerabilities
in smart contracts using graph representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import json
import os
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SolidityASTParser:
    """Parser for converting Solidity code to Abstract Syntax Tree"""
    
    def __init__(self):
        self.node_types = {
            'contract': 0, 'function': 1, 'variable': 2, 'modifier': 3,
            'event': 4, 'struct': 5, 'enum': 6, 'mapping': 7, 'array': 8,
            'if': 9, 'for': 10, 'while': 11, 'require': 12, 'assert': 13,
            'call': 14, 'transfer': 15, 'send': 16, 'delegatecall': 17,
            'selfdestruct': 18, 'suicide': 19, 'assembly': 20, 'modifier_call': 21
        }
    
    def parse_contract(self, source_code: str) -> nx.DiGraph:
        """Parse Solidity contract to AST graph"""
        G = nx.DiGraph()
        
        # Add root node
        G.add_node(0, type='contract', name='root', line=0)
        
        # Simple parsing (in practice, use a proper Solidity parser)
        lines = source_code.split('\n')
        node_id = 1
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//') or line.startswith('/*'):
                continue
            
            # Detect different constructs
            if 'contract' in line and '{' in line:
                G.add_node(node_id, type='contract', name=self._extract_name(line), line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif 'function' in line and '(' in line:
                G.add_node(node_id, type='function', name=self._extract_name(line), line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif 'modifier' in line:
                G.add_node(node_id, type='modifier', name=self._extract_name(line), line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif 'event' in line:
                G.add_node(node_id, type='event', name=self._extract_name(line), line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif 'require(' in line:
                G.add_node(node_id, type='require', name='require', line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif 'assert(' in line:
                G.add_node(node_id, type='assert', name='assert', line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif '.call(' in line:
                G.add_node(node_id, type='call', name='call', line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif '.transfer(' in line:
                G.add_node(node_id, type='transfer', name='transfer', line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif '.send(' in line:
                G.add_node(node_id, type='send', name='send', line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
            elif 'selfdestruct(' in line:
                G.add_node(node_id, type='selfdestruct', name='selfdestruct', line=i+1)
                G.add_edge(0, node_id)
                node_id += 1
        
        return G
    
    def _extract_name(self, line: str) -> str:
        """Extract name from code line"""
        # Simple name extraction
        words = line.split()
        for i, word in enumerate(words):
            if word in ['contract', 'function', 'modifier', 'event']:
                if i + 1 < len(words):
                    return words[i + 1].split('(')[0].split('{')[0]
        return 'unknown'

class SmartContractGraphDataset:
    """Dataset class for graph-based smart contract vulnerability detection"""
    
    def __init__(self, graphs: List[nx.DiGraph], labels: List[int]):
        self.graphs = graphs
        self.labels = labels
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        label = self.labels[idx]
        
        # Convert NetworkX graph to PyTorch Geometric Data
        edge_index = torch.tensor(list(graph.edges()), dtype=torch.long).t().contiguous()
        
        # Node features (one-hot encoding of node types)
        node_features = []
        for node in graph.nodes():
            node_type = graph.nodes[node].get('type', 'unknown')
            # Create one-hot encoding
            feature = [0] * len(self.node_types)
            if node_type in self.node_types:
                feature[self.node_types[node_type]] = 1
            else:
                feature[0] = 1  # Default to contract type
            node_features.append(feature)
        
        x = torch.tensor(node_features, dtype=torch.float)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, y=torch.tensor(label, dtype=torch.long))
        
        return data

class GNNVulnerabilityDetector(nn.Module):
    """GNN model for vulnerability detection"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 gnn_type: str = 'GCN'):
        super(GNNVulnerabilityDetector, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # Input layer
        if gnn_type == 'GCN':
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=4, dropout=dropout))
        elif gnn_type == 'SAGE':
            self.gnn_layers.append(SAGEConv(input_dim, hidden_dim))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if gnn_type == 'GCN':
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
            elif gnn_type == 'GAT':
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout))
            elif gnn_type == 'SAGE':
                self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Output layer
        if gnn_type == 'GCN':
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == 'GAT':
            self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout))
        elif gnn_type == 'SAGE':
            self.gnn_layers.append(SAGEConv(hidden_dim, hidden_dim))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean and max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, data):
        """Forward pass through the model"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            if i < len(self.gnn_layers) - 1:  # Don't apply activation to last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        logits = self.classifier(x)
        
        return logits

class GNNTrainer:
    """Trainer class for GNN vulnerability detection"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 2,
                 num_layers: int = 3,
                 dropout: float = 0.1,
                 gnn_type: str = 'GCN',
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 device: str = "auto"):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.gnn_type = gnn_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = GNNVulnerabilityDetector(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            gnn_type=gnn_type
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Initialize parser
        self.parser = SolidityASTParser()
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def prepare_dataset(self, 
                       source_codes: List[str], 
                       labels: List[int],
                       split_ratio: float = 0.8) -> Tuple[SmartContractGraphDataset, SmartContractGraphDataset]:
        """Prepare train and validation datasets"""
        
        # Parse source codes to graphs
        graphs = []
        for code in source_codes:
            try:
                graph = self.parser.parse_contract(code)
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to parse contract: {e}")
                # Create empty graph as fallback
                empty_graph = nx.DiGraph()
                empty_graph.add_node(0, type='contract', name='empty')
                graphs.append(empty_graph)
        
        # Split data
        train_graphs, val_graphs, train_labels, val_labels = train_test_split(
            graphs, labels, test_size=1-split_ratio, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = SmartContractGraphDataset(train_graphs, train_labels)
        val_dataset = SmartContractGraphDataset(val_graphs, val_labels)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            # Forward pass
            logits = self.model(batch)
            loss = F.cross_entropy(logits, batch.y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                logits = self.model(batch)
                loss = F.cross_entropy(logits, batch.y)
                
                # Statistics
                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='weighted'
        )
        
        return avg_loss, accuracy, f1
    
    def train(self, 
              train_dataset: SmartContractGraphDataset,
              val_dataset: SmartContractGraphDataset,
              num_epochs: int = 100,
              batch_size: int = 32,
              patience: int = 10):
        """Train the model"""
        logger.info("Starting GNN training...")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1 = self.evaluate(val_loader)
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['val_f1'].append(val_f1)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/gnn_best_model.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log progress
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        logger.info("Training completed!")
        logger.info(f"Best validation F1: {best_val_f1:.4f}")
        
        return self.training_history
    
    def evaluate_model(self, test_dataset: SmartContractGraphDataset) -> Dict[str, float]:
        """Evaluate the model on test dataset"""
        logger.info("Evaluating GNN model...")
        
        # Load best model
        self.model.load_state_dict(torch.load('models/gnn_best_model.pth'))
        
        # Create test loader
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        test_loss, test_acc, test_f1 = self.evaluate(test_loader)
        
        metrics = {
            'accuracy': test_acc,
            'f1_score': test_f1,
            'loss': test_loss
        }
        
        logger.info(f"Test metrics: {metrics}")
        
        return metrics
    
    def predict(self, source_codes: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new source codes"""
        # Parse source codes to graphs
        graphs = []
        for code in source_codes:
            try:
                graph = self.parser.parse_contract(code)
                graphs.append(graph)
            except Exception as e:
                logger.warning(f"Failed to parse contract: {e}")
                empty_graph = nx.DiGraph()
                empty_graph.add_node(0, type='contract', name='empty')
                graphs.append(empty_graph)
        
        # Create dataset
        dataset = SmartContractGraphDataset(graphs, [0] * len(graphs))
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Load best model
        self.model.load_state_dict(torch.load('models/gnn_best_model.pth'))
        self.model.eval()
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                logits = self.model(batch)
                probs = F.softmax(logits, dim=1)
                
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_probs)
    
    def save_model(self, path: str):
        """Save the trained model"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), f"{path}/model.pth")
        
        # Save configuration
        config = {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'gnn_type': self.gnn_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save training history
        with open(f"{path}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        # Load configuration
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
        
        # Update model parameters
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.output_dim = config['output_dim']
        self.num_layers = config['num_layers']
        self.dropout = config['dropout']
        self.gnn_type = config['gnn_type']
        
        # Recreate model
        self.model = GNNVulnerabilityDetector(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            dropout=self.dropout,
            gnn_type=self.gnn_type
        ).to(self.device)
        
        # Load model state
        self.model.load_state_dict(torch.load(f"{path}/model.pth"))
        
        logger.info(f"Model loaded from {path}")

def train_gnn_model(data_path: str, 
                   output_dir: str = "models/gnn_output",
                   gnn_type: str = 'GCN',
                   num_epochs: int = 100,
                   batch_size: int = 32,
                   learning_rate: float = 0.001):
    """Train GNN model for vulnerability detection"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare data
    source_codes = df['source_code'].tolist()
    labels = df['vulnerability_type_encoded'].tolist()
    
    # Initialize trainer
    trainer = GNNTrainer(
        input_dim=22,  # Number of node types
        gnn_type=gnn_type,
        learning_rate=learning_rate
    )
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_dataset(source_codes, labels)
    
    # Train model
    history = trainer.train(train_dataset, val_dataset, num_epochs, batch_size)
    
    # Save model
    trainer.save_model(output_dir)
    
    return trainer, history

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train GNN model for vulnerability detection')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', default='models/gnn_output', help='Output directory')
    parser.add_argument('--gnn_type', default='GCN', choices=['GCN', 'GAT', 'SAGE'], help='GNN type')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    trainer, history = train_gnn_model(
        data_path=args.data,
        output_dir=args.output,
        gnn_type=args.gnn_type,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("GNN training completed!")
