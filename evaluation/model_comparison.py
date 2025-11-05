#!/usr/bin/env python3
"""
Model Comparison and Benchmarking Framework

This module provides comprehensive model comparison and benchmarking
capabilities for smart contract vulnerability detection models.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import torch
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Add models directory to path
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

from codebert_model import CodeBERTTrainer
from gnn_model import GNNTrainer
from metrics import VulnerabilityDetectionMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelComparisonFramework:
    """Comprehensive model comparison and benchmarking framework"""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "evaluation/comparison_results",
                 config_path: Optional[str] = None):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize metrics
        self.metrics = VulnerabilityDetectionMetrics()
        
        # Results storage
        self.comparison_results = {
            'models': {},
            'benchmarks': {},
            'statistical_tests': {},
            'recommendations': {},
            'metadata': {
                'data_path': data_path,
                'output_dir': str(output_dir),
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load comparison configuration"""
        default_config = {
            'models': {
                'codebert': {
                    'model_name': 'microsoft/codebert-base',
                    'num_epochs': 5,
                    'batch_size': 16,
                    'learning_rate': 2e-5
                },
                'gnn': {
                    'gnn_type': 'GCN',
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'num_epochs': 50,
                    'batch_size': 32,
                    'learning_rate': 0.001
                }
            },
            'evaluation': {
                'cv_folds': 5,
                'test_size': 0.2,
                'random_state': 42,
                'metrics': ['accuracy', 'precision', 'recall', 'f1']
            },
            'benchmarks': {
                'baseline_models': ['random', 'majority'],
                'statistical_tests': True,
                'significance_level': 0.05
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
    
    def load_data(self) -> pd.DataFrame:
        """Load and prepare data for comparison"""
        logger.info("Loading data for model comparison...")
        
        df = pd.read_csv(self.data_path)
        
        # Data validation
        required_columns = ['source_code', 'vulnerability_type_encoded']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        df = df.dropna(subset=['source_code', 'vulnerability_type_encoded'])
        
        logger.info(f"Loaded {len(df)} samples for comparison")
        logger.info(f"Vulnerability distribution: {df['vulnerability_type'].value_counts().to_dict()}")
        
        return df
    
    def train_codebert_model(self, df: pd.DataFrame) -> Dict:
        """Train and evaluate CodeBERT model"""
        logger.info("Training CodeBERT model...")
        
        try:
            # Prepare data
            texts = df['source_code'].tolist()
            labels = df['vulnerability_type_encoded'].tolist()
            
            # Initialize trainer
            trainer = CodeBERTTrainer(
                model_name=self.config['models']['codebert']['model_name'],
                num_epochs=self.config['models']['codebert']['num_epochs'],
                batch_size=self.config['models']['codebert']['batch_size'],
                learning_rate=self.config['models']['codebert']['learning_rate']
            )
            
            # Prepare datasets
            train_dataset, val_dataset = trainer.prepare_dataset(texts, labels)
            
            # Train model
            train_result = trainer.train(train_dataset, val_dataset, 
                                      f"{self.output_dir}/codebert")
            
            # Evaluate
            val_metrics = trainer.evaluate(val_dataset)
            
            # Cross-validation
            cv_scores = self._cross_validate_codebert(texts, labels)
            
            results = {
                'model_type': 'CodeBERT',
                'training_metrics': train_result.metrics,
                'validation_metrics': val_metrics,
                'cv_scores': cv_scores,
                'model_path': f"{self.output_dir}/codebert",
                'config': self.config['models']['codebert']
            }
            
            self.comparison_results['models']['codebert'] = results
            logger.info("CodeBERT training completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"CodeBERT training failed: {e}")
            return {'error': str(e)}
    
    def train_gnn_model(self, df: pd.DataFrame) -> Dict:
        """Train and evaluate GNN model"""
        logger.info("Training GNN model...")
        
        try:
            # Prepare data
            source_codes = df['source_code'].tolist()
            labels = df['vulnerability_type_encoded'].tolist()
            
            # Initialize trainer
            trainer = GNNTrainer(
                input_dim=22,
                hidden_dim=self.config['models']['gnn']['hidden_dim'],
                num_layers=self.config['models']['gnn']['num_layers'],
                gnn_type=self.config['models']['gnn']['gnn_type'],
                learning_rate=self.config['models']['gnn']['learning_rate']
            )
            
            # Prepare datasets
            train_dataset, val_dataset = trainer.prepare_dataset(source_codes, labels)
            
            # Train model
            history = trainer.train(
                train_dataset, val_dataset,
                num_epochs=self.config['models']['gnn']['num_epochs'],
                batch_size=self.config['models']['gnn']['batch_size'],
                patience=10
            )
            
            # Evaluate
            val_metrics = trainer.evaluate_model(val_dataset)
            
            # Cross-validation
            cv_scores = self._cross_validate_gnn(source_codes, labels)
            
            results = {
                'model_type': 'GNN',
                'training_history': history,
                'validation_metrics': val_metrics,
                'cv_scores': cv_scores,
                'model_path': f"{self.output_dir}/gnn",
                'config': self.config['models']['gnn']
            }
            
            self.comparison_results['models']['gnn'] = results
            logger.info("GNN training completed successfully")
            
            return results
            
        except Exception as e:
            logger.error(f"GNN training failed: {e}")
            return {'error': str(e)}
    
    def _cross_validate_codebert(self, texts: List[str], labels: List[int]) -> Dict:
        """Cross-validate CodeBERT model"""
        logger.info("Cross-validating CodeBERT model...")
        
        try:
            cv_folds = self.config['evaluation']['cv_folds']
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                random_state=self.config['evaluation']['random_state'])
            
            cv_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
                logger.info(f"CodeBERT CV fold {fold + 1}/{cv_folds}")
                
                # Split data
                train_texts = [texts[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_texts = [texts[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]
                
                # Train model
                trainer = CodeBERTTrainer(
                    model_name=self.config['models']['codebert']['model_name'],
                    num_epochs=2,  # Reduced for CV
                    batch_size=self.config['models']['codebert']['batch_size'],
                    learning_rate=self.config['models']['codebert']['learning_rate']
                )
                
                train_dataset, _ = trainer.prepare_dataset(train_texts, train_labels, split_ratio=1.0)
                val_dataset, _ = trainer.prepare_dataset(val_texts, val_labels, split_ratio=1.0)
                
                # Train and evaluate
                trainer.setup_trainer(train_dataset, val_dataset, 
                                   f"{self.output_dir}/cv/codebert_fold_{fold}")
                trainer.train(train_dataset, val_dataset, 
                            f"{self.output_dir}/cv/codebert_fold_{fold}")
                metrics = trainer.evaluate(val_dataset)
                
                # Store scores
                cv_scores['accuracy'].append(metrics.get('accuracy', 0))
                cv_scores['f1'].append(metrics.get('f1', 0))
                cv_scores['precision'].append(metrics.get('precision', 0))
                cv_scores['recall'].append(metrics.get('recall', 0))
            
            # Calculate statistics
            cv_stats = {}
            for metric, scores in cv_scores.items():
                cv_stats[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
            
            return cv_stats
            
        except Exception as e:
            logger.warning(f"CodeBERT cross-validation failed: {e}")
            return {'error': str(e)}
    
    def _cross_validate_gnn(self, source_codes: List[str], labels: List[int]) -> Dict:
        """Cross-validate GNN model"""
        logger.info("Cross-validating GNN model...")
        
        try:
            cv_folds = self.config['evaluation']['cv_folds']
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                random_state=self.config['evaluation']['random_state'])
            
            cv_scores = {'accuracy': [], 'f1_score': [], 'precision': [], 'recall': []}
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(source_codes, labels)):
                logger.info(f"GNN CV fold {fold + 1}/{cv_folds}")
                
                # Split data
                train_codes = [source_codes[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_codes = [source_codes[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]
                
                # Train model
                trainer = GNNTrainer(
                    input_dim=22,
                    hidden_dim=self.config['models']['gnn']['hidden_dim'],
                    num_layers=self.config['models']['gnn']['num_layers'],
                    gnn_type=self.config['models']['gnn']['gnn_type'],
                    learning_rate=self.config['models']['gnn']['learning_rate']
                )
                
                train_dataset, _ = trainer.prepare_dataset(train_codes, train_labels, split_ratio=1.0)
                val_dataset, _ = trainer.prepare_dataset(val_codes, val_labels, split_ratio=1.0)
                
                # Train and evaluate
                history = trainer.train(train_dataset, val_dataset, 
                                      num_epochs=10,  # Reduced for CV
                                      batch_size=self.config['models']['gnn']['batch_size'],
                                      patience=3)
                metrics = trainer.evaluate_model(val_dataset)
                
                # Store scores
                cv_scores['accuracy'].append(metrics.get('accuracy', 0))
                cv_scores['f1_score'].append(metrics.get('f1_score', 0))
                cv_scores['precision'].append(metrics.get('precision', 0))
                cv_scores['recall'].append(metrics.get('recall', 0))
            
            # Calculate statistics
            cv_stats = {}
            for metric, scores in cv_scores.items():
                cv_stats[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
            
            return cv_stats
            
        except Exception as e:
            logger.warning(f"GNN cross-validation failed: {e}")
            return {'error': str(e)}
    
    def run_baseline_benchmarks(self, df: pd.DataFrame) -> Dict:
        """Run baseline model benchmarks"""
        logger.info("Running baseline benchmarks...")
        
        labels = df['vulnerability_type_encoded'].values
        n_samples = len(labels)
        
        baseline_results = {}
        
        # Random baseline
        if 'random' in self.config['benchmarks']['baseline_models']:
            np.random.seed(42)
            random_preds = np.random.randint(0, len(np.unique(labels)), n_samples)
            
            baseline_results['random'] = {
                'accuracy': accuracy_score(labels, random_preds),
                'f1': f1_score(labels, random_preds, average='weighted', zero_division=0),
                'precision': precision_score(labels, random_preds, average='weighted', zero_division=0),
                'recall': recall_score(labels, random_preds, average='weighted', zero_division=0)
            }
        
        # Majority class baseline
        if 'majority' in self.config['benchmarks']['baseline_models']:
            majority_class = np.bincount(labels).argmax()
            majority_preds = np.full(n_samples, majority_class)
            
            baseline_results['majority'] = {
                'accuracy': accuracy_score(labels, majority_preds),
                'f1': f1_score(labels, majority_preds, average='weighted', zero_division=0),
                'precision': precision_score(labels, majority_preds, average='weighted', zero_division=0),
                'recall': recall_score(labels, majority_preds, average='weighted', zero_division=0)
            }
        
        self.comparison_results['benchmarks'] = baseline_results
        logger.info("Baseline benchmarks completed")
        
        return baseline_results
    
    def perform_statistical_tests(self) -> Dict:
        """Perform statistical significance tests"""
        logger.info("Performing statistical significance tests...")
        
        if not self.config['benchmarks']['statistical_tests']:
            return {}
        
        statistical_tests = {}
        
        # Compare models if both are available
        if 'codebert' in self.comparison_results['models'] and 'gnn' in self.comparison_results['models']:
            codebert_cv = self.comparison_results['models']['codebert'].get('cv_scores', {})
            gnn_cv = self.comparison_results['models']['gnn'].get('cv_scores', {})
            
            if 'f1' in codebert_cv and 'f1_score' in gnn_cv:
                # Perform t-test for F1 scores
                from scipy import stats
                
                codebert_f1_scores = codebert_cv['f1']['scores']
                gnn_f1_scores = gnn_cv['f1_score']['scores']
                
                t_stat, p_value = stats.ttest_rel(codebert_f1_scores, gnn_f1_scores)
                
                statistical_tests['codebert_vs_gnn'] = {
                    'test': 'paired_t_test',
                    'metric': 'f1_score',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config['benchmarks']['significance_level'],
                    'codebert_mean': np.mean(codebert_f1_scores),
                    'gnn_mean': np.mean(gnn_f1_scores)
                }
        
        self.comparison_results['statistical_tests'] = statistical_tests
        
        return statistical_tests
    
    def generate_comparison_report(self) -> Dict:
        """Generate comprehensive comparison report"""
        logger.info("Generating comparison report...")
        
        # Model performance comparison
        model_comparison = self._compare_model_performance()
        
        # Efficiency comparison
        efficiency_comparison = self._compare_model_efficiency()
        
        # Recommendations
        recommendations = self._generate_recommendations()
        
        # Generate visualizations
        self._generate_comparison_plots()
        
        # Create comprehensive report
        report = {
            'model_comparison': model_comparison,
            'efficiency_comparison': efficiency_comparison,
            'recommendations': recommendations,
            'statistical_tests': self.comparison_results['statistical_tests'],
            'baseline_benchmarks': self.comparison_results['benchmarks']
        }
        
        # Save report
        with open(self.output_dir / "comparison_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        logger.info("Comparison report generated successfully")
        
        return report
    
    def _compare_model_performance(self) -> Dict:
        """Compare model performance metrics"""
        comparison = {}
        
        for model_name, model_results in self.comparison_results['models'].items():
            if 'error' not in model_results:
                comparison[model_name] = {
                    'validation_metrics': model_results.get('validation_metrics', {}),
                    'cv_performance': self._extract_cv_performance(model_results.get('cv_scores', {}))
                }
        
        return comparison
    
    def _extract_cv_performance(self, cv_scores: Dict) -> Dict:
        """Extract cross-validation performance metrics"""
        performance = {}
        
        for metric, stats in cv_scores.items():
            if isinstance(stats, dict) and 'mean' in stats:
                performance[metric] = {
                    'mean': stats['mean'],
                    'std': stats['std']
                }
        
        return performance
    
    def _compare_model_efficiency(self) -> Dict:
        """Compare model efficiency metrics"""
        efficiency = {}
        
        # This would include metrics like:
        # - Training time
        # - Inference time
        # - Model size
        # - Memory usage
        
        # Placeholder implementation
        efficiency['codebert'] = {
            'training_time': 'N/A',
            'inference_time': 'N/A',
            'model_size': 'N/A'
        }
        
        efficiency['gnn'] = {
            'training_time': 'N/A',
            'inference_time': 'N/A',
            'model_size': 'N/A'
        }
        
        return efficiency
    
    def _generate_recommendations(self) -> List[str]:
        """Generate model recommendations"""
        recommendations = []
        
        # Compare model performance
        if 'codebert' in self.comparison_results['models'] and 'gnn' in self.comparison_results['models']:
            codebert_f1 = self.comparison_results['models']['codebert'].get('validation_metrics', {}).get('f1', 0)
            gnn_f1 = self.comparison_results['models']['gnn'].get('validation_metrics', {}).get('f1_score', 0)
            
            if codebert_f1 > gnn_f1:
                recommendations.append("CodeBERT shows better overall performance")
            else:
                recommendations.append("GNN shows better overall performance")
        
        # Statistical significance
        if self.comparison_results['statistical_tests']:
            for test_name, test_results in self.comparison_results['statistical_tests'].items():
                if test_results.get('significant', False):
                    recommendations.append(f"Statistical significance found in {test_name}")
        
        # General recommendations
        recommendations.extend([
            "Consider ensemble methods for improved performance",
            "Collect more data for underrepresented classes",
            "Fine-tune hyperparameters for better results",
            "Implement data augmentation techniques"
        ])
        
        return recommendations
    
    def _generate_comparison_plots(self):
        """Generate comparison visualization plots"""
        try:
            # Model performance comparison
            if len(self.comparison_results['models']) > 1:
                self._plot_model_comparison()
            
            # Cross-validation comparison
            if any('cv_scores' in model for model in self.comparison_results['models'].values()):
                self._plot_cv_comparison()
            
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        models = []
        f1_scores = []
        
        for model_name, model_results in self.comparison_results['models'].items():
            if 'error' not in model_results:
                models.append(model_name.upper())
                f1_scores.append(model_results.get('validation_metrics', {}).get('f1', 
                                   model_results.get('validation_metrics', {}).get('f1_score', 0)))
        
        if models:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, f1_scores, alpha=0.7, color=['skyblue', 'lightcoral'])
            plt.ylabel('F1 Score')
            plt.title('Model Performance Comparison')
            plt.ylim(0, 1)
            
            # Add value labels on bars
            for bar, score in zip(bars, f1_scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_cv_comparison(self):
        """Plot cross-validation comparison"""
        plt.figure(figsize=(12, 8))
        
        models = []
        means = []
        stds = []
        
        for model_name, model_results in self.comparison_results['models'].items():
            if 'error' not in model_results and 'cv_scores' in model_results:
                cv_scores = model_results['cv_scores']
                if 'f1' in cv_scores or 'f1_score' in cv_scores:
                    f1_key = 'f1' if 'f1' in cv_scores else 'f1_score'
                    models.append(model_name.upper())
                    means.append(cv_scores[f1_key]['mean'])
                    stds.append(cv_scores[f1_key]['std'])
        
        if models:
            x_pos = np.arange(len(models))
            bars = plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
                          color=['skyblue', 'lightcoral'])
            plt.xlabel('Models')
            plt.ylabel('F1 Score')
            plt.title('Cross-Validation Performance Comparison')
            plt.xticks(x_pos, models)
            plt.ylim(0, 1)
            
            # Add value labels
            for i, (mean, std) in enumerate(zip(means, stds)):
                plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}', 
                        ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "cv_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _generate_markdown_report(self, report: Dict):
        """Generate markdown comparison report"""
        md_content = f"""# Model Comparison Report

## Overview
- **Comparison Date**: {datetime.now().isoformat()}
- **Models Compared**: {', '.join(self.comparison_results['models'].keys())}

## Model Performance Comparison

### Validation Metrics
"""
        
        for model_name, model_data in self.comparison_results['models'].items():
            if 'error' not in model_data:
                md_content += f"""
#### {model_name.upper()}
- **F1 Score**: {model_data.get('validation_metrics', {}).get('f1', model_data.get('validation_metrics', {}).get('f1_score', 'N/A')):.4f}
- **Accuracy**: {model_data.get('validation_metrics', {}).get('accuracy', 'N/A'):.4f}
- **Precision**: {model_data.get('validation_metrics', {}).get('precision', 'N/A'):.4f}
- **Recall**: {model_data.get('validation_metrics', {}).get('recall', 'N/A'):.4f}
"""
        
        # Cross-validation results
        md_content += "\n## Cross-Validation Results\n"
        for model_name, model_data in self.comparison_results['models'].items():
            if 'error' not in model_data and 'cv_scores' in model_data:
                cv_scores = model_data['cv_scores']
                if 'f1' in cv_scores or 'f1_score' in cv_scores:
                    f1_key = 'f1' if 'f1' in cv_scores else 'f1_score'
                    md_content += f"""
#### {model_name.upper()}
- **CV F1 Score**: {cv_scores[f1_key]['mean']:.4f} ± {cv_scores[f1_key]['std']:.4f}
"""
        
        # Statistical tests
        if self.comparison_results['statistical_tests']:
            md_content += "\n## Statistical Significance Tests\n"
            for test_name, test_results in self.comparison_results['statistical_tests'].items():
                md_content += f"""
#### {test_name}
- **Test**: {test_results.get('test', 'N/A')}
- **P-value**: {test_results.get('p_value', 'N/A'):.4f}
- **Significant**: {test_results.get('significant', 'N/A')}
"""
        
        # Recommendations
        md_content += "\n## Recommendations\n"
        for rec in report.get('recommendations', []):
            md_content += f"- {rec}\n"
        
        # Save report
        with open(self.output_dir / "comparison_report.md", "w") as f:
            f.write(md_content)
    
    def run_full_comparison(self):
        """Run complete model comparison pipeline"""
        logger.info("Starting full model comparison pipeline...")
        
        # Load data
        df = self.load_data()
        
        # Train models
        logger.info("Training CodeBERT model...")
        self.train_codebert_model(df)
        
        logger.info("Training GNN model...")
        self.train_gnn_model(df)
        
        # Run baseline benchmarks
        logger.info("Running baseline benchmarks...")
        self.run_baseline_benchmarks(df)
        
        # Perform statistical tests
        logger.info("Performing statistical tests...")
        self.perform_statistical_tests()
        
        # Generate comparison report
        logger.info("Generating comparison report...")
        report = self.generate_comparison_report()
        
        # Save complete results
        with open(self.output_dir / "complete_comparison_results.json", "w") as f:
            json.dump(self.comparison_results, f, indent=2, default=str)
        
        logger.info("Model comparison completed successfully!")
        logger.info(f"Results saved to {self.output_dir}")
        
        return self.comparison_results

def main():
    """Main function for model comparison"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare vulnerability detection models')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', default='evaluation/comparison_results', help='Output directory')
    parser.add_argument('--config', help='Path to comparison configuration JSON')
    
    args = parser.parse_args()
    
    # Initialize comparison framework
    framework = ModelComparisonFramework(
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config
    )
    
    # Run comparison
    results = framework.run_full_comparison()
    
    print("Model comparison completed successfully!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
