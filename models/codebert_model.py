#!/usr/bin/env python3
"""
CodeBERT Model for Smart Contract Vulnerability Detection

This module implements a CodeBERT-based model for detecting vulnerabilities
in smart contracts using transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import json
import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartContractDataset:
    """Dataset class for smart contract vulnerability detection"""
    
    def __init__(self, 
                 texts: List[str], 
                 labels: List[int], 
                 tokenizer,
                 max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class CodeBERTVulnerabilityDetector(nn.Module):
    """CodeBERT model for vulnerability detection"""
    
    def __init__(self, 
                 model_name: str = "microsoft/codebert-base",
                 num_classes: int = 2,
                 dropout_rate: float = 0.1,
                 freeze_encoder: bool = False):
        super(CodeBERTVulnerabilityDetector, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Load pre-trained CodeBERT model
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights"""
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        torch.nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass through the model"""
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions
        }

class CodeBERTTrainer:
    """Trainer class for CodeBERT vulnerability detection"""
    
    def __init__(self, 
                 model_name: str = "microsoft/codebert-base",
                 num_classes: int = 2,
                 max_length: int = 512,
                 batch_size: int = 16,
                 learning_rate: float = 2e-5,
                 num_epochs: int = 10,
                 warmup_steps: int = 500,
                 weight_decay: float = 0.01,
                 device: str = "auto"):
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = CodeBERTVulnerabilityDetector(
            model_name=model_name,
            num_classes=num_classes
        ).to(self.device)
        
        # Initialize trainer
        self.trainer = None
        self.training_history = []
    
    def prepare_dataset(self, 
                       texts: List[str], 
                       labels: List[int],
                       split_ratio: float = 0.8) -> Tuple[SmartContractDataset, SmartContractDataset]:
        """Prepare train and validation datasets"""
        from sklearn.model_selection import train_test_split
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=1-split_ratio, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = SmartContractDataset(
            train_texts, train_labels, self.tokenizer, self.max_length
        )
        val_dataset = SmartContractDataset(
            val_texts, val_labels, self.tokenizer, self.max_length
        )
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def setup_trainer(self, 
                     train_dataset: SmartContractDataset,
                     val_dataset: SmartContractDataset,
                     output_dir: str = "models/codebert_output"):
        """Setup Hugging Face trainer"""
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to="none",  # Disable wandb for now
            seed=42,
            data_seed=42,
            remove_unused_columns=False
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
    
    def _compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(self, 
              train_dataset: SmartContractDataset,
              val_dataset: SmartContractDataset,
              output_dir: str = "models/codebert_output"):
        """Train the model"""
        logger.info("Starting CodeBERT training...")
        
        # Setup trainer
        self.setup_trainer(train_dataset, val_dataset, output_dir)
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training history
        self.training_history = train_result.metrics
        with open(f"{output_dir}/training_history.json", "w") as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info("Training completed!")
        logger.info(f"Training metrics: {self.training_history}")
        
        return train_result
    
    def evaluate(self, 
                 test_dataset: SmartContractDataset) -> Dict[str, float]:
        """Evaluate the model on test dataset"""
        logger.info("Evaluating CodeBERT model...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        
        # Extract metrics
        metrics = predictions.metrics
        
        logger.info(f"Evaluation metrics: {metrics}")
        
        return metrics
    
    def predict(self, 
                texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new texts"""
        # Create dataset
        dataset = SmartContractDataset(
            texts, [0] * len(texts), self.tokenizer, self.max_length
        )
        
        # Get predictions
        predictions = self.trainer.predict(dataset)
        logits = predictions.predictions
        
        # Get predicted classes and probabilities
        predicted_classes = np.argmax(logits, axis=1)
        predicted_probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        
        return predicted_classes, predicted_probs
    
    def save_model(self, path: str):
        """Save the trained model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save model configuration
        config = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs
        }
        
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model"""
        self.model = CodeBERTVulnerabilityDetector.from_pretrained(path)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        # Load configuration
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
        
        logger.info(f"Model loaded from {path}")

def train_codebert_model(data_path: str, 
                         output_dir: str = "models/codebert_output",
                         model_name: str = "microsoft/codebert-base",
                         num_epochs: int = 10,
                         batch_size: int = 16,
                         learning_rate: float = 2e-5):
    """Train CodeBERT model for vulnerability detection"""
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Prepare data
    texts = df['source_code'].tolist()
    labels = df['vulnerability_type_encoded'].tolist()
    
    # Initialize trainer
    trainer = CodeBERTTrainer(
        model_name=model_name,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Prepare datasets
    train_dataset, val_dataset = trainer.prepare_dataset(texts, labels)
    
    # Train model
    train_result = trainer.train(train_dataset, val_dataset, output_dir)
    
    # Save model
    trainer.save_model(output_dir)
    
    return trainer, train_result

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CodeBERT model for vulnerability detection')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', default='models/codebert_output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    trainer, result = train_codebert_model(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("CodeBERT training completed!")
