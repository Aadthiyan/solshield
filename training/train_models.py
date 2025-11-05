#!/usr/bin/env python3
"""
Model Training Pipeline

This script provides a comprehensive training pipeline for both CodeBERT and GNN models
for smart contract vulnerability detection.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / "models"))

from codebert_model import CodeBERTTrainer, train_codebert_model
from gnn_model import GNNTrainer, train_gnn_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """Comprehensive training pipeline for vulnerability detection models"""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "training/output",
                 config_path: Optional[str] = None):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize results storage
        self.results = {
            'codebert': {},
            'gnn': {},
            'comparison': {},
            'metadata': {
                'data_path': data_path,
                'output_dir': str(output_dir),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load training configuration"""
        default_config = {
            'codebert': {
                'model_name': 'microsoft/codebert-base',
                'num_epochs': 10,
                'batch_size': 16,
                'learning_rate': 2e-5,
                'max_length': 512,
                'warmup_steps': 500,
                'weight_decay': 0.01
            },
            'gnn': {
                'gnn_type': 'GCN',
                'hidden_dim': 64,
                'num_layers': 3,
                'dropout': 0.1,
                'num_epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'patience': 10
            },
            'evaluation': {
                'test_size': 0.2,
                'val_size': 0.1,
                'random_state': 42,
                'cv_folds': 5
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configurations
            for key in default_config:
                if key in user_config:
                    default_config[key].update(user_config[key])
        
        return default_config
    
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, Dict]:
        """Load and prepare data for training"""
        logger.info("Loading and preparing data...")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Data preparation
        data_info = {
            'total_samples': len(df),
            'features': list(df.columns),
            'vulnerability_distribution': df['vulnerability_type'].value_counts().to_dict(),
            'severity_distribution': df['severity'].value_counts().to_dict(),
            'source_distribution': df['data_source'].value_counts().to_dict()
        }
        
        # Check for required columns
        required_columns = ['source_code', 'vulnerability_type_encoded']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Handle missing values
        df = df.dropna(subset=['source_code', 'vulnerability_type_encoded'])
        
        logger.info(f"Loaded {len(df)} samples")
        logger.info(f"Vulnerability distribution: {data_info['vulnerability_distribution']}")
        
        return df, data_info
    
    def train_codebert(self, df: pd.DataFrame) -> Dict:
        """Train CodeBERT model"""
        logger.info("Training CodeBERT model...")
        
        # Prepare data
        texts = df['source_code'].tolist()
        labels = df['vulnerability_type_encoded'].tolist()
        
        # Initialize trainer
        trainer = CodeBERTTrainer(
            model_name=self.config['codebert']['model_name'],
            num_epochs=self.config['codebert']['num_epochs'],
            batch_size=self.config['codebert']['batch_size'],
            learning_rate=self.config['codebert']['learning_rate']
        )
        
        # Prepare datasets
        train_dataset, val_dataset = trainer.prepare_dataset(texts, labels)
        
        # Train model
        train_result = trainer.train(train_dataset, val_dataset, 
                                   f"{self.output_dir}/codebert")
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate(val_dataset)
        
        # Save results
        codebert_results = {
            'model_path': f"{self.output_dir}/codebert",
            'training_metrics': train_result.metrics,
            'validation_metrics': val_metrics,
            'config': self.config['codebert']
        }
        
        self.results['codebert'] = codebert_results
        
        logger.info("CodeBERT training completed!")
        logger.info(f"Validation metrics: {val_metrics}")
        
        return codebert_results
    
    def train_gnn(self, df: pd.DataFrame) -> Dict:
        """Train GNN model"""
        logger.info("Training GNN model...")
        
        # Prepare data
        source_codes = df['source_code'].tolist()
        labels = df['vulnerability_type_encoded'].tolist()
        
        # Initialize trainer
        trainer = GNNTrainer(
            input_dim=22,  # Number of node types
            hidden_dim=self.config['gnn']['hidden_dim'],
            num_layers=self.config['gnn']['num_layers'],
            dropout=self.config['gnn']['dropout'],
            gnn_type=self.config['gnn']['gnn_type'],
            learning_rate=self.config['gnn']['learning_rate'],
            weight_decay=self.config['gnn']['weight_decay']
        )
        
        # Prepare datasets
        train_dataset, val_dataset = trainer.prepare_dataset(source_codes, labels)
        
        # Train model
        history = trainer.train(
            train_dataset, val_dataset,
            num_epochs=self.config['gnn']['num_epochs'],
            batch_size=self.config['gnn']['batch_size'],
            patience=self.config['gnn']['patience']
        )
        
        # Evaluate on validation set
        val_metrics = trainer.evaluate_model(val_dataset)
        
        # Save results
        gnn_results = {
            'model_path': f"{self.output_dir}/gnn",
            'training_history': history,
            'validation_metrics': val_metrics,
            'config': self.config['gnn']
        }
        
        self.results['gnn'] = gnn_results
        
        logger.info("GNN training completed!")
        logger.info(f"Validation metrics: {val_metrics}")
        
        return gnn_results
    
    def cross_validate_models(self, df: pd.DataFrame) -> Dict:
        """Perform cross-validation for both models"""
        logger.info("Performing cross-validation...")
        
        # Prepare data
        texts = df['source_code'].tolist()
        labels = df['vulnerability_type_encoded'].tolist()
        
        # Cross-validation setup
        cv_folds = self.config['evaluation']['cv_folds']
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                            random_state=self.config['evaluation']['random_state'])
        
        cv_results = {
            'codebert': {'scores': [], 'folds': []},
            'gnn': {'scores': [], 'folds': []}
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            logger.info(f"Cross-validation fold {fold + 1}/{cv_folds}")
            
            # Split data
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # CodeBERT cross-validation
            try:
                codebert_trainer = CodeBERTTrainer(
                    model_name=self.config['codebert']['model_name'],
                    num_epochs=3,  # Reduced for CV
                    batch_size=self.config['codebert']['batch_size'],
                    learning_rate=self.config['codebert']['learning_rate']
                )
                
                train_dataset, _ = codebert_trainer.prepare_dataset(train_texts, train_labels, split_ratio=1.0)
                val_dataset, _ = codebert_trainer.prepare_dataset(val_texts, val_labels, split_ratio=1.0)
                
                # Train and evaluate
                codebert_trainer.setup_trainer(train_dataset, val_dataset, 
                                             f"{self.output_dir}/cv/codebert_fold_{fold}")
                codebert_trainer.train(train_dataset, val_dataset, 
                                     f"{self.output_dir}/cv/codebert_fold_{fold}")
                codebert_metrics = codebert_trainer.evaluate(val_dataset)
                
                cv_results['codebert']['scores'].append(codebert_metrics['f1'])
                cv_results['codebert']['folds'].append({
                    'fold': fold,
                    'metrics': codebert_metrics
                })
                
            except Exception as e:
                logger.warning(f"CodeBERT fold {fold} failed: {e}")
                cv_results['codebert']['scores'].append(0.0)
            
            # GNN cross-validation
            try:
                gnn_trainer = GNNTrainer(
                    input_dim=22,
                    hidden_dim=self.config['gnn']['hidden_dim'],
                    num_layers=self.config['gnn']['num_layers'],
                    dropout=self.config['gnn']['dropout'],
                    gnn_type=self.config['gnn']['gnn_type'],
                    learning_rate=self.config['gnn']['learning_rate']
                )
                
                train_dataset, _ = gnn_trainer.prepare_dataset(train_texts, train_labels, split_ratio=1.0)
                val_dataset, _ = gnn_trainer.prepare_dataset(val_texts, val_labels, split_ratio=1.0)
                
                # Train and evaluate
                history = gnn_trainer.train(train_dataset, val_dataset, 
                                          num_epochs=20,  # Reduced for CV
                                          batch_size=self.config['gnn']['batch_size'],
                                          patience=5)
                gnn_metrics = gnn_trainer.evaluate_model(val_dataset)
                
                cv_results['gnn']['scores'].append(gnn_metrics['f1_score'])
                cv_results['gnn']['folds'].append({
                    'fold': fold,
                    'metrics': gnn_metrics
                })
                
            except Exception as e:
                logger.warning(f"GNN fold {fold} failed: {e}")
                cv_results['gnn']['scores'].append(0.0)
        
        # Calculate CV statistics
        cv_results['codebert']['mean_f1'] = np.mean(cv_results['codebert']['scores'])
        cv_results['codebert']['std_f1'] = np.std(cv_results['codebert']['scores'])
        cv_results['gnn']['mean_f1'] = np.mean(cv_results['gnn']['scores'])
        cv_results['gnn']['std_f1'] = np.std(cv_results['gnn']['scores'])
        
        logger.info(f"CodeBERT CV F1: {cv_results['codebert']['mean_f1']:.4f} ± {cv_results['codebert']['std_f1']:.4f}")
        logger.info(f"GNN CV F1: {cv_results['gnn']['mean_f1']:.4f} ± {cv_results['gnn']['std_f1']:.4f}")
        
        return cv_results
    
    def compare_models(self, test_df: pd.DataFrame) -> Dict:
        """Compare model performance on test set"""
        logger.info("Comparing model performance...")
        
        # Load trained models
        codebert_path = f"{self.output_dir}/codebert"
        gnn_path = f"{self.output_dir}/gnn"
        
        comparison_results = {
            'test_metrics': {},
            'model_comparison': {},
            'statistical_tests': {}
        }
        
        # Test CodeBERT
        try:
            codebert_trainer = CodeBERTTrainer()
            codebert_trainer.load_model(codebert_path)
            
            test_texts = test_df['source_code'].tolist()
            test_labels = test_df['vulnerability_type_encoded'].tolist()
            
            # Create test dataset
            test_dataset = codebert_trainer.prepare_dataset(test_texts, test_labels, split_ratio=1.0)[0]
            
            # Evaluate
            codebert_metrics = codebert_trainer.evaluate(test_dataset)
            comparison_results['test_metrics']['codebert'] = codebert_metrics
            
        except Exception as e:
            logger.warning(f"CodeBERT evaluation failed: {e}")
            comparison_results['test_metrics']['codebert'] = {}
        
        # Test GNN
        try:
            gnn_trainer = GNNTrainer(input_dim=22)
            gnn_trainer.load_model(gnn_path)
            
            test_source_codes = test_df['source_code'].tolist()
            test_labels = test_df['vulnerability_type_encoded'].tolist()
            
            # Create test dataset
            test_dataset = gnn_trainer.prepare_dataset(test_source_codes, test_labels, split_ratio=1.0)[0]
            
            # Evaluate
            gnn_metrics = gnn_trainer.evaluate_model(test_dataset)
            comparison_results['test_metrics']['gnn'] = gnn_metrics
            
        except Exception as e:
            logger.warning(f"GNN evaluation failed: {e}")
            comparison_results['test_metrics']['gnn'] = {}
        
        # Model comparison
        if 'codebert' in comparison_results['test_metrics'] and 'gnn' in comparison_results['test_metrics']:
            codebert_f1 = comparison_results['test_metrics']['codebert'].get('f1', 0)
            gnn_f1 = comparison_results['test_metrics']['gnn'].get('f1_score', 0)
            
            comparison_results['model_comparison'] = {
                'best_model': 'codebert' if codebert_f1 > gnn_f1 else 'gnn',
                'f1_difference': abs(codebert_f1 - gnn_f1),
                'codebert_f1': codebert_f1,
                'gnn_f1': gnn_f1
            }
        
        self.results['comparison'] = comparison_results
        
        return comparison_results
    
    def generate_reports(self, data_info: Dict, cv_results: Dict, comparison_results: Dict):
        """Generate comprehensive training reports"""
        logger.info("Generating training reports...")
        
        # Create reports directory
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Training summary report
        summary_report = {
            'data_info': data_info,
            'model_results': {
                'codebert': self.results['codebert'],
                'gnn': self.results['gnn']
            },
            'cross_validation': cv_results,
            'model_comparison': comparison_results,
            'recommendations': self._generate_recommendations()
        }
        
        # Save summary report
        with open(reports_dir / "training_summary.json", "w") as f:
            json.dump(summary_report, f, indent=2, default=str)
        
        # Generate visualizations
        self._generate_visualizations(reports_dir, cv_results, comparison_results)
        
        # Generate markdown report
        self._generate_markdown_report(reports_dir, summary_report)
        
        logger.info(f"Reports generated in {reports_dir}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate model recommendations based on results"""
        recommendations = []
        
        # Check if models were trained successfully
        if self.results['codebert'] and self.results['gnn']:
            codebert_f1 = self.results['codebert'].get('validation_metrics', {}).get('f1', 0)
            gnn_f1 = self.results['gnn'].get('validation_metrics', {}).get('f1_score', 0)
            
            if codebert_f1 > gnn_f1:
                recommendations.append("CodeBERT shows better performance for this dataset")
            else:
                recommendations.append("GNN shows better performance for this dataset")
        
        recommendations.extend([
            "Consider ensemble methods for improved performance",
            "Collect more data for underrepresented vulnerability types",
            "Fine-tune hyperparameters for better results",
            "Implement data augmentation techniques"
        ])
        
        return recommendations
    
    def _generate_visualizations(self, reports_dir: Path, cv_results: Dict, comparison_results: Dict):
        """Generate visualization plots"""
        try:
            # Cross-validation comparison plot
            if cv_results and 'codebert' in cv_results and 'gnn' in cv_results:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                models = ['CodeBERT', 'GNN']
                means = [cv_results['codebert']['mean_f1'], cv_results['gnn']['mean_f1']]
                stds = [cv_results['codebert']['std_f1'], cv_results['gnn']['std_f1']]
                
                ax.bar(models, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_ylabel('F1 Score')
                ax.set_title('Cross-Validation Performance Comparison')
                ax.set_ylim(0, 1)
                
                plt.tight_layout()
                plt.savefig(reports_dir / "cv_comparison.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Model performance comparison
            if comparison_results and 'test_metrics' in comparison_results:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                models = []
                f1_scores = []
                
                for model, metrics in comparison_results['test_metrics'].items():
                    if metrics:
                        models.append(model.upper())
                        f1_scores.append(metrics.get('f1', metrics.get('f1_score', 0)))
                
                if models:
                    ax.bar(models, f1_scores, alpha=0.7)
                    ax.set_ylabel('F1 Score')
                    ax.set_title('Test Set Performance Comparison')
                    ax.set_ylim(0, 1)
                    
                    plt.tight_layout()
                    plt.savefig(reports_dir / "test_comparison.png", dpi=300, bbox_inches='tight')
                    plt.close()
        
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
    
    def _generate_markdown_report(self, reports_dir: Path, summary_report: Dict):
        """Generate markdown training report"""
        md_content = f"""# Model Training Report

## Overview
- **Training Date**: {summary_report['data_info'].get('timestamp', 'Unknown')}
- **Total Samples**: {summary_report['data_info'].get('total_samples', 'Unknown')}
- **Models Trained**: CodeBERT, GNN

## Data Information
- **Vulnerability Distribution**: {summary_report['data_info'].get('vulnerability_distribution', {})}
- **Severity Distribution**: {summary_report['data_info'].get('severity_distribution', {})}
- **Source Distribution**: {summary_report['data_info'].get('source_distribution', {})}

## Model Results

### CodeBERT
- **Validation F1**: {summary_report['model_results']['codebert'].get('validation_metrics', {}).get('f1', 'N/A')}
- **Validation Accuracy**: {summary_report['model_results']['codebert'].get('validation_metrics', {}).get('accuracy', 'N/A')}

### GNN
- **Validation F1**: {summary_report['model_results']['gnn'].get('validation_metrics', {}).get('f1_score', 'N/A')}
- **Validation Accuracy**: {summary_report['model_results']['gnn'].get('validation_metrics', {}).get('accuracy', 'N/A')}

## Cross-Validation Results
- **CodeBERT CV F1**: {summary_report['cross_validation'].get('codebert', {}).get('mean_f1', 'N/A')} ± {summary_report['cross_validation'].get('codebert', {}).get('std_f1', 'N/A')}
- **GNN CV F1**: {summary_report['cross_validation'].get('gnn', {}).get('mean_f1', 'N/A')} ± {summary_report['cross_validation'].get('gnn', {}).get('std_f1', 'N/A')}

## Recommendations
{chr(10).join(f"- {rec}" for rec in summary_report.get('recommendations', []))}

## Files Generated
- `training_summary.json`: Complete training results
- `cv_comparison.png`: Cross-validation comparison plot
- `test_comparison.png`: Test set performance comparison
"""
        
        with open(reports_dir / "training_report.md", "w") as f:
            f.write(md_content)
    
    def run_full_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("Starting full training pipeline...")
        
        # Load and prepare data
        df, data_info = self.load_and_prepare_data()
        
        # Split data for final evaluation
        train_df, test_df = train_test_split(
            df, test_size=self.config['evaluation']['test_size'],
            random_state=self.config['evaluation']['random_state'],
            stratify=df['vulnerability_type_encoded']
        )
        
        # Train models
        logger.info("Training CodeBERT model...")
        self.train_codebert(train_df)
        
        logger.info("Training GNN model...")
        self.train_gnn(train_df)
        
        # Cross-validation
        logger.info("Performing cross-validation...")
        cv_results = self.cross_validate_models(train_df)
        
        # Model comparison
        logger.info("Comparing models...")
        comparison_results = self.compare_models(test_df)
        
        # Generate reports
        logger.info("Generating reports...")
        self.generate_reports(data_info, cv_results, comparison_results)
        
        # Save complete results
        with open(self.output_dir / "complete_results.json", "w") as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Results saved to {self.output_dir}")
        
        return self.results

def main():
    """Main function for training pipeline"""
    parser = argparse.ArgumentParser(description='Train vulnerability detection models')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', default='training/output', help='Output directory')
    parser.add_argument('--config', help='Path to training configuration JSON')
    parser.add_argument('--models', nargs='+', choices=['codebert', 'gnn', 'all'], 
                       default=['all'], help='Models to train')
    parser.add_argument('--cv', action='store_true', help='Perform cross-validation')
    parser.add_argument('--compare', action='store_true', help='Compare model performance')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ModelTrainingPipeline(
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config
    )
    
    # Run pipeline
    results = pipeline.run_full_pipeline()
    
    print("Training completed successfully!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
