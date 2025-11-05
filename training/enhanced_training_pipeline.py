#!/usr/bin/env python3
"""
Enhanced Training Pipeline with Advanced Features

This module implements the enhanced training pipeline that incorporates:
1. Joint syntax-semantic graph learning
2. Proxy labeling for data augmentation
3. Adversarial training for robustness
4. Comprehensive evaluation metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm
import wandb
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Import our advanced modules
from models.joint_graph_learning import HierarchicalSyntaxSemanticGNN, JointGraphDataProcessor
from data.proxy_labeling import ProxyLabelGenerator, DataAugmentationWithProxyLabels
from models.adversarial_defense import AdversarialTrainingPipeline, RobustnessEvaluator
from models.codebert_model import CodeBERTModel
from models.gnn_model import GNNModel

class EnhancedDataset(Dataset):
    """Enhanced dataset with proxy labels and adversarial samples"""
    
    def __init__(self, contracts: List[Dict[str, Any]], 
                 use_proxy_labels: bool = True,
                 use_adversarial: bool = True,
                 augmentation_factor: int = 2):
        self.contracts = contracts
        self.use_proxy_labels = use_proxy_labels
        self.use_adversarial = use_adversarial
        self.augmentation_factor = augmentation_factor
        
        # Initialize processors
        self.graph_processor = JointGraphDataProcessor()
        self.proxy_generator = ProxyLabelGenerator()
        self.data_augmenter = DataAugmentationWithProxyLabels()
        
        # Process dataset
        self.processed_contracts = self._process_contracts()
    
    def _process_contracts(self) -> List[Dict[str, Any]]:
        """Process contracts with advanced features"""
        processed = []
        
        for contract in tqdm(self.contracts, desc="Processing contracts"):
            # Generate proxy labels
            if self.use_proxy_labels:
                proxy_labels = self.proxy_generator.generate_proxy_labels(
                    contract['code'], contract['label']
                )
                contract['proxy_labels'] = proxy_labels
            
            # Generate adversarial samples
            if self.use_adversarial:
                adversarial_samples = self._generate_adversarial_samples(contract)
                contract['adversarial_samples'] = adversarial_samples
            
            # Convert to graph format
            graph_data = self.graph_processor.process_contract(
                contract['code'], contract['label']
            )
            contract['graph_data'] = graph_data
            
            processed.append(contract)
        
        # Apply data augmentation
        if self.augmentation_factor > 0:
            augmented_contracts = self.data_augmenter.augment_dataset(
                processed, self.augmentation_factor
            )
            processed.extend(augmented_contracts)
        
        return processed
    
    def _generate_adversarial_samples(self, contract: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adversarial samples for a contract"""
        # This would use the adversarial generators
        # For now, return empty list
        return []
    
    def __len__(self):
        return len(self.processed_contracts)
    
    def __getitem__(self, idx):
        contract = self.processed_contracts[idx]
        return {
            'graph_data': contract['graph_data'],
            'label': contract['label'],
            'proxy_labels': contract.get('proxy_labels', {}),
            'adversarial_samples': contract.get('adversarial_samples', []),
            'contract_id': contract.get('id', idx)
        }

class EnhancedModel(nn.Module):
    """Enhanced model combining all advanced features"""
    
    def __init__(self, 
                 joint_gnn_config: Dict[str, Any],
                 codebert_config: Dict[str, Any],
                 gnn_config: Dict[str, Any],
                 fusion_config: Dict[str, Any]):
        super().__init__()
        
        # Initialize component models
        self.joint_gnn = HierarchicalSyntaxSemanticGNN(**joint_gnn_config)
        self.codebert = CodeBERTModel(**codebert_config)
        self.gnn = GNNModel(**gnn_config)
        
        # Fusion layers
        self.fusion_layers = nn.ModuleList([
            nn.Linear(fusion_config['input_dim'], fusion_config['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(fusion_config['dropout']),
            nn.Linear(fusion_config['hidden_dim'], fusion_config['output_dim'])
        ])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(fusion_config['output_dim'], 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for model fusion
        self.attention = nn.MultiheadAttention(
            embed_dim=fusion_config['output_dim'],
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
    
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass through enhanced model"""
        # Get outputs from each model
        joint_outputs = self.joint_gnn(
            batch['joint_x'], batch['joint_edge_index'], 
            batch.get('joint_edge_attr', None), batch['joint_batch']
        )
        
        codebert_outputs = self.codebert(
            batch['codebert_input_ids'], 
            batch['codebert_attention_mask']
        )
        
        gnn_outputs = self.gnn(
            batch['gnn_x'], batch['gnn_edge_index'], 
            batch.get('gnn_edge_attr', None), batch['gnn_batch']
        )
        
        # Fuse model outputs
        model_outputs = torch.stack([
            joint_outputs['graph_embedding'],
            codebert_outputs['pooler_output'],
            gnn_outputs['graph_embedding']
        ], dim=1)  # [batch_size, 3, embedding_dim]
        
        # Apply attention fusion
        fused_output, attention_weights = self.attention(
            model_outputs, model_outputs, model_outputs
        )
        fused_output = fused_output.mean(dim=1)  # [batch_size, embedding_dim]
        
        # Apply fusion layers
        for layer in self.fusion_layers:
            fused_output = layer(fused_output)
        
        # Final classification
        vulnerability_prediction = self.classifier(fused_output)
        
        return {
            'vulnerability_prediction': vulnerability_prediction,
            'joint_embeddings': joint_outputs['node_embeddings'],
            'codebert_embeddings': codebert_outputs['last_hidden_state'],
            'gnn_embeddings': gnn_outputs['node_embeddings'],
            'fused_embeddings': fused_output,
            'attention_weights': attention_weights,
            'model_confidence': self._calculate_confidence(vulnerability_prediction)
        }
    
    def _calculate_confidence(self, predictions: torch.Tensor) -> torch.Tensor:
        """Calculate prediction confidence"""
        # Distance from decision boundary (0.5)
        confidence = torch.abs(predictions - 0.5) * 2
        return confidence

class EnhancedTrainingPipeline:
    """Enhanced training pipeline with all advanced features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Initialize models
        self.model = EnhancedModel(
            joint_gnn_config=config['joint_gnn'],
            codebert_config=config['codebert'],
            gnn_config=config['gnn'],
            fusion_config=config['fusion']
        ).to(self.device)
        
        # Initialize training components
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs']
        )
        
        # Initialize adversarial training
        self.adversarial_pipeline = AdversarialTrainingPipeline(
            self.model, self.device
        )
        
        # Initialize robustness evaluator
        self.robustness_evaluator = RobustnessEvaluator(
            self.model, self.device
        )
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize wandb if enabled
        if config.get('use_wandb', False):
            wandb.init(
                project=config['project_name'],
                config=config,
                name=f"enhanced_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('enhanced_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """Train the enhanced model"""
        self.logger.info("Starting enhanced training pipeline")
        
        best_val_accuracy = 0.0
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'robustness_scores': []
        }
        
        for epoch in range(self.config['epochs']):
            self.logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            
            # Robustness evaluation
            if epoch % self.config.get('robustness_eval_frequency', 5) == 0:
                robustness_metrics = self.robustness_evaluator.evaluate_robustness(val_loader)
                training_history['robustness_scores'].append(robustness_metrics)
                self.logger.info(f"Robustness metrics: {robustness_metrics}")
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self._log_metrics(epoch, train_metrics, val_metrics)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self._save_model(epoch, val_metrics)
            
            # Update training history
            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            training_history['train_accuracy'].append(train_metrics['accuracy'])
            training_history['val_accuracy'].append(val_metrics['accuracy'])
        
        self.logger.info("Training completed")
        return training_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Calculate loss
            loss = self._calculate_loss(outputs, batch)
            
            # Adversarial training
            if self.config.get('use_adversarial_training', True):
                adversarial_metrics = self.adversarial_pipeline.adversarial_training_step(
                    batch, self.optimizer
                )
                loss += adversarial_metrics['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clipping', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']
                )
            
            self.optimizer.step()
            
            # Calculate metrics
            predictions = (outputs['vulnerability_prediction'] > 0.5).float()
            accuracy = (predictions.squeeze() == batch['labels']).float().mean()
            
            total_loss += loss.item()
            total_accuracy += accuracy.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Accuracy': f"{accuracy.item():.4f}"
            })
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': total_accuracy / num_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = self._move_batch_to_device(batch)
                
                outputs = self.model(batch)
                loss = self._calculate_loss(outputs, batch)
                
                predictions = (outputs['vulnerability_prediction'] > 0.5).float()
                accuracy = (predictions.squeeze() == batch['labels']).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='binary'
        )
        
        try:
            auc = roc_auc_score(all_labels, all_predictions)
        except ValueError:
            auc = 0.0
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': total_accuracy / len(val_loader),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def _calculate_loss(self, outputs: Dict[str, torch.Tensor], 
                       batch: Dict[str, Any]) -> torch.Tensor:
        """Calculate combined loss"""
        # Primary loss
        primary_loss = F.binary_cross_entropy(
            outputs['vulnerability_prediction'], 
            batch['labels'].unsqueeze(1).float()
        )
        
        # Confidence loss (encourage high confidence)
        confidence_loss = -torch.mean(outputs['model_confidence'])
        
        # Attention regularization
        attention_weights = outputs.get('attention_weights', None)
        attention_loss = 0.0
        if attention_weights is not None:
            attention_loss = torch.mean(torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1))
        
        # Combined loss
        total_loss = (
            primary_loss + 
            0.1 * confidence_loss + 
            0.05 * attention_loss
        )
        
        return total_loss
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_metrics(self, epoch: int, train_metrics: Dict[str, float], 
                    val_metrics: Dict[str, float]):
        """Log training metrics"""
        self.logger.info(
            f"Epoch {epoch + 1} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val F1: {val_metrics['f1']:.4f}, "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )
        
        # Log to wandb
        if self.config.get('use_wandb', False):
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc']
            })
    
    def _save_model(self, epoch: int, metrics: Dict[str, float]):
        """Save the best model"""
        model_path = f"enhanced_model_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, model_path)
        
        self.logger.info(f"Model saved: {model_path}")
    
    def evaluate_robustness(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model robustness"""
        self.logger.info("Evaluating model robustness")
        
        robustness_metrics = self.robustness_evaluator.evaluate_robustness(test_loader)
        
        self.logger.info(f"Robustness evaluation completed: {robustness_metrics}")
        return robustness_metrics

def create_enhanced_config() -> Dict[str, Any]:
    """Create configuration for enhanced training"""
    return {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'epochs': 100,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'gradient_clipping': 1.0,
        'use_adversarial_training': True,
        'use_proxy_labels': True,
        'robustness_eval_frequency': 5,
        'use_wandb': False,
        'project_name': 'smart-contract-vulnerability-detection',
        
        'joint_gnn': {
            'input_dim': 128,
            'hidden_dim': 256,
            'output_dim': 64,
            'num_layers': 4,
            'num_heads': 8,
            'dropout': 0.1
        },
        
        'codebert': {
            'model_name': 'microsoft/codebert-base',
            'num_labels': 1,
            'dropout': 0.1
        },
        
        'gnn': {
            'input_dim': 128,
            'hidden_dim': 256,
            'output_dim': 64,
            'num_layers': 3,
            'dropout': 0.1
        },
        
        'fusion': {
            'input_dim': 192,  # 64 * 3
            'hidden_dim': 256,
            'output_dim': 128,
            'dropout': 0.2
        }
    }

# Example usage
if __name__ == "__main__":
    # Create enhanced configuration
    config = create_enhanced_config()
    
    # Initialize enhanced training pipeline
    pipeline = EnhancedTrainingPipeline(config)
    
    print("Enhanced training pipeline initialized successfully!")
    print(f"Device: {config['device']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Adversarial training: {config['use_adversarial_training']}")
    print(f"Proxy labels: {config['use_proxy_labels']}")
