#!/usr/bin/env python3
"""
Cross-Validation with Benchmark Datasets

This module implements comprehensive cross-validation for smart contract
vulnerability detection models using benchmark datasets.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import torch
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Add project directories to path
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

from codebert_model import CodeBERTTrainer
from gnn_model import GNNTrainer
from metrics import VulnerabilityDetectionMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossValidationFramework:
    """Comprehensive cross-validation framework for vulnerability detection models"""
    
    def __init__(self, 
                 data_path: str,
                 output_dir: str = "benchmarks/cv_results",
                 config_path: Optional[str] = None):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize metrics
        self.metrics = VulnerabilityDetectionMetrics()
        
        # Results storage
        self.cv_results = {
            'codebert': {},
            'gnn': {},
            'baseline': {},
            'statistical_analysis': {},
            'metadata': {
                'data_path': data_path,
                'output_dir': str(output_dir),
                'timestamp': datetime.now().isoformat(),
                'cv_folds': self.config['cv_folds'],
                'random_state': self.config['random_state']
            }
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load cross-validation configuration"""
        default_config = {
            'cv_folds': 5,
            'random_state': 42,
            'test_size': 0.2,
            'models': {
                'codebert': {
                    'model_name': 'microsoft/codebert-base',
                    'num_epochs': 3,  # Reduced for CV
                    'batch_size': 16,
                    'learning_rate': 2e-5,
                    'max_length': 512
                },
                'gnn': {
                    'gnn_type': 'GCN',
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'num_epochs': 20,  # Reduced for CV
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'patience': 5
                }
            },
            'baseline_models': ['random', 'majority'],
            'statistical_tests': True,
            'significance_level': 0.05
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge configurations
            for key in default_config:
                if key in user_config:
                    if isinstance(default_config[key], dict):
                        default_config[key].update(user_config[key])
                    else:
                        default_config[key] = user_config[key]
        
        return default_config
    
    def load_benchmark_data(self) -> pd.DataFrame:
        """Load and prepare benchmark data for cross-validation"""
        logger.info("Loading benchmark data for cross-validation...")
        
        df = pd.read_csv(self.data_path)
        
        # Data validation
        required_columns = ['source_code', 'vulnerability_type_encoded']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Clean data
        df = df.dropna(subset=['source_code', 'vulnerability_type_encoded'])
        
        # Check class distribution
        class_distribution = df['vulnerability_type_encoded'].value_counts()
        logger.info(f"Class distribution: {class_distribution.to_dict()}")
        
        # Check minimum samples per class
        min_samples = class_distribution.min()
        if min_samples < self.config['cv_folds']:
            logger.warning(f"Some classes have fewer samples than CV folds ({min_samples} < {self.config['cv_folds']})")
        
        logger.info(f"Loaded {len(df)} samples for cross-validation")
        
        return df
    
    def cross_validate_codebert(self, df: pd.DataFrame) -> Dict:
        """Cross-validate CodeBERT model"""
        logger.info("Cross-validating CodeBERT model...")
        
        # Prepare data
        texts = df['source_code'].tolist()
        labels = df['vulnerability_type_encoded'].tolist()
        
        # Initialize cross-validation
        skf = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'fold_details': []
        }
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(texts, labels)):
            logger.info(f"CodeBERT CV fold {fold + 1}/{self.config['cv_folds']}")
            
            try:
                # Split data
                train_texts = [texts[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_texts = [texts[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]
                
                # Initialize trainer
                trainer = CodeBERTTrainer(
                    model_name=self.config['models']['codebert']['model_name'],
                    num_epochs=self.config['models']['codebert']['num_epochs'],
                    batch_size=self.config['models']['codebert']['batch_size'],
                    learning_rate=self.config['models']['codebert']['learning_rate']
                )
                
                # Prepare datasets
                train_dataset, _ = trainer.prepare_dataset(train_texts, train_labels, split_ratio=1.0)
                val_dataset, _ = trainer.prepare_dataset(val_texts, val_labels, split_ratio=1.0)
                
                # Train model
                trainer.setup_trainer(train_dataset, val_dataset, 
                                   f"{self.output_dir}/codebert_fold_{fold}")
                trainer.train(train_dataset, val_dataset, 
                            f"{self.output_dir}/codebert_fold_{fold}")
                
                # Evaluate
                metrics = trainer.evaluate(val_dataset)
                
                # Store results
                fold_result = {
                    'fold': fold,
                    'metrics': metrics,
                    'train_size': len(train_dataset),
                    'val_size': len(val_dataset)
                }
                fold_results.append(fold_result)
                
                # Store scores
                cv_scores['accuracy'].append(metrics.get('accuracy', 0))
                cv_scores['precision'].append(metrics.get('precision', 0))
                cv_scores['recall'].append(metrics.get('recall', 0))
                cv_scores['f1'].append(metrics.get('f1', 0))
                
            except Exception as e:
                logger.error(f"CodeBERT fold {fold} failed: {e}")
                cv_scores['accuracy'].append(0)
                cv_scores['precision'].append(0)
                cv_scores['recall'].append(0)
                cv_scores['f1'].append(0)
                fold_results.append({'fold': fold, 'error': str(e)})
        
        # Calculate statistics
        cv_stats = {}
        for metric, scores in cv_scores.items():
            if metric != 'fold_details':
                cv_stats[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores
                }
        
        cv_stats['fold_details'] = fold_results
        
        self.cv_results['codebert'] = cv_stats
        
        logger.info(f"CodeBERT CV completed - F1: {cv_stats['f1']['mean']:.4f} ± {cv_stats['f1']['std']:.4f}")
        
        return cv_stats
    
    def cross_validate_gnn(self, df: pd.DataFrame) -> Dict:
        """Cross-validate GNN model"""
        logger.info("Cross-validating GNN model...")
        
        # Prepare data
        source_codes = df['source_code'].tolist()
        labels = df['vulnerability_type_encoded'].tolist()
        
        # Initialize cross-validation
        skf = StratifiedKFold(
            n_splits=self.config['cv_folds'],
            shuffle=True,
            random_state=self.config['random_state']
        )
        
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'fold_details': []
        }
        
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(source_codes, labels)):
            logger.info(f"GNN CV fold {fold + 1}/{self.config['cv_folds']}")
            
            try:
                # Split data
                train_codes = [source_codes[i] for i in train_idx]
                train_labels = [labels[i] for i in train_idx]
                val_codes = [source_codes[i] for i in val_idx]
                val_labels = [labels[i] for i in val_idx]
                
                # Initialize trainer
                trainer = GNNTrainer(
                    input_dim=22,
                    hidden_dim=self.config['models']['gnn']['hidden_dim'],
                    num_layers=self.config['models']['gnn']['num_layers'],
                    gnn_type=self.config['models']['gnn']['gnn_type'],
                    learning_rate=self.config['models']['gnn']['learning_rate']
                )
                
                # Prepare datasets
                train_dataset, _ = trainer.prepare_dataset(train_codes, train_labels, split_ratio=1.0)
                val_dataset, _ = trainer.prepare_dataset(val_codes, val_labels, split_ratio=1.0)
                
                # Train model
                history = trainer.train(
                    train_dataset, val_dataset,
                    num_epochs=self.config['models']['gnn']['num_epochs'],
                    batch_size=self.config['models']['gnn']['batch_size'],
                    patience=self.config['models']['gnn']['patience']
                )
                
                # Evaluate
                metrics = trainer.evaluate_model(val_dataset)
                
                # Store results
                fold_result = {
                    'fold': fold,
                    'metrics': metrics,
                    'training_history': history,
                    'train_size': len(train_dataset),
                    'val_size': len(val_dataset)
                }
                fold_results.append(fold_result)
                
                # Store scores
                cv_scores['accuracy'].append(metrics.get('accuracy', 0))
                cv_scores['precision'].append(metrics.get('precision', 0))
                cv_scores['recall'].append(metrics.get('recall', 0))
                cv_scores['f1_score'].append(metrics.get('f1_score', 0))
                
            except Exception as e:
                logger.error(f"GNN fold {fold} failed: {e}")
                cv_scores['accuracy'].append(0)
                cv_scores['precision'].append(0)
                cv_scores['recall'].append(0)
                cv_scores['f1_score'].append(0)
                fold_results.append({'fold': fold, 'error': str(e)})
        
        # Calculate statistics
        cv_stats = {}
        for metric, scores in cv_scores.items():
            if metric != 'fold_details':
                cv_stats[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'scores': scores
                }
        
        cv_stats['fold_details'] = fold_results
        
        self.cv_results['gnn'] = cv_stats
        
        logger.info(f"GNN CV completed - F1: {cv_stats['f1_score']['mean']:.4f} ± {cv_stats['f1_score']['std']:.4f}")
        
        return cv_stats
    
    def run_baseline_cross_validation(self, df: pd.DataFrame) -> Dict:
        """Run cross-validation for baseline models"""
        logger.info("Running baseline model cross-validation...")
        
        labels = df['vulnerability_type_encoded'].values
        n_samples = len(labels)
        
        baseline_results = {}
        
        # Random baseline
        if 'random' in self.config['baseline_models']:
            logger.info("Cross-validating random baseline...")
            
            random_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
            
            for fold in range(self.config['cv_folds']):
                np.random.seed(self.config['random_state'] + fold)
                random_preds = np.random.randint(0, len(np.unique(labels)), n_samples)
                
                random_scores['accuracy'].append(accuracy_score(labels, random_preds))
                random_scores['f1'].append(f1_score(labels, random_preds, average='weighted', zero_division=0))
                random_scores['precision'].append(precision_score(labels, random_preds, average='weighted', zero_division=0))
                random_scores['recall'].append(recall_score(labels, random_preds, average='weighted', zero_division=0))
            
            # Calculate statistics
            baseline_results['random'] = {}
            for metric, scores in random_scores.items():
                baseline_results['random'][metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
        
        # Majority class baseline
        if 'majority' in self.config['baseline_models']:
            logger.info("Cross-validating majority baseline...")
            
            majority_class = np.bincount(labels).argmax()
            majority_preds = np.full(n_samples, majority_class)
            
            majority_scores = {
                'accuracy': accuracy_score(labels, majority_preds),
                'f1': f1_score(labels, majority_preds, average='weighted', zero_division=0),
                'precision': precision_score(labels, majority_preds, average='weighted', zero_division=0),
                'recall': recall_score(labels, majority_preds, average='weighted', zero_division=0)
            }
            
            baseline_results['majority'] = {}
            for metric, score in majority_scores.items():
                baseline_results['majority'][metric] = {
                    'mean': score,
                    'std': 0.0,
                    'scores': [score]
                }
        
        self.cv_results['baseline'] = baseline_results
        
        logger.info("Baseline cross-validation completed")
        
        return baseline_results
    
    def perform_statistical_analysis(self) -> Dict:
        """Perform statistical analysis of cross-validation results"""
        logger.info("Performing statistical analysis...")
        
        if not self.config['statistical_tests']:
            return {}
        
        statistical_analysis = {}
        
        # Compare models if both are available
        if 'codebert' in self.cv_results and 'gnn' in self.cv_results:
            codebert_f1_scores = self.cv_results['codebert']['f1']['scores']
            gnn_f1_scores = self.cv_results['gnn']['f1_score']['scores']
            
            # Perform paired t-test
            from scipy import stats
            
            t_stat, p_value = stats.ttest_rel(codebert_f1_scores, gnn_f1_scores)
            
            statistical_analysis['codebert_vs_gnn'] = {
                'test': 'paired_t_test',
                'metric': 'f1_score',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.config['significance_level'],
                'codebert_mean': np.mean(codebert_f1_scores),
                'gnn_mean': np.mean(gnn_f1_scores),
                'effect_size': np.mean(codebert_f1_scores) - np.mean(gnn_f1_scores)
            }
        
        # Compare with baselines
        if 'codebert' in self.cv_results and 'baseline' in self.cv_results:
            codebert_f1_scores = self.cv_results['codebert']['f1']['scores']
            
            for baseline_name, baseline_results in self.cv_results['baseline'].items():
                baseline_f1_scores = baseline_results['f1']['scores']
                
                if len(baseline_f1_scores) > 1:  # Only for random baseline
                    t_stat, p_value = stats.ttest_rel(codebert_f1_scores, baseline_f1_scores)
                else:  # For majority baseline
                    # Use one-sample t-test
                    t_stat, p_value = stats.ttest_1samp(codebert_f1_scores, baseline_f1_scores[0])
                
                statistical_analysis[f'codebert_vs_{baseline_name}'] = {
                    'test': 'paired_t_test' if len(baseline_f1_scores) > 1 else 'one_sample_t_test',
                    'metric': 'f1_score',
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < self.config['significance_level'],
                    'codebert_mean': np.mean(codebert_f1_scores),
                    'baseline_mean': np.mean(baseline_f1_scores),
                    'effect_size': np.mean(codebert_f1_scores) - np.mean(baseline_f1_scores)
                }
        
        self.cv_results['statistical_analysis'] = statistical_analysis
        
        logger.info("Statistical analysis completed")
        
        return statistical_analysis
    
    def generate_cv_report(self) -> Dict:
        """Generate comprehensive cross-validation report"""
        logger.info("Generating cross-validation report...")
        
        # Create reports directory
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Generate summary statistics
        summary_stats = self._generate_summary_statistics()
        
        # Generate visualizations
        self._generate_cv_visualizations()
        
        # Generate markdown report
        self._generate_markdown_report(summary_stats)
        
        # Save complete results
        with open(self.output_dir / "cv_results.json", "w") as f:
            json.dump(self.cv_results, f, indent=2, default=str)
        
        logger.info("Cross-validation report generated successfully")
        
        return summary_stats
    
    def _generate_summary_statistics(self) -> Dict:
        """Generate summary statistics for all models"""
        summary = {
            'model_performance': {},
            'best_model': None,
            'statistical_significance': {},
            'recommendations': []
        }
        
        # Model performance summary
        for model_name, model_results in self.cv_results.items():
            if model_name in ['codebert', 'gnn'] and model_results:
                f1_key = 'f1' if 'f1' in model_results else 'f1_score'
                if f1_key in model_results:
                    summary['model_performance'][model_name] = {
                        'f1_mean': model_results[f1_key]['mean'],
                        'f1_std': model_results[f1_key]['std'],
                        'accuracy_mean': model_results['accuracy']['mean'],
                        'accuracy_std': model_results['accuracy']['std']
                    }
        
        # Determine best model
        if summary['model_performance']:
            best_model = max(summary['model_performance'].items(), 
                            key=lambda x: x[1]['f1_mean'])
            summary['best_model'] = best_model[0]
        
        # Statistical significance
        if 'statistical_analysis' in self.cv_results:
            summary['statistical_significance'] = self.cv_results['statistical_analysis']
        
        # Generate recommendations
        summary['recommendations'] = self._generate_recommendations()
        
        return summary
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on cross-validation results"""
        recommendations = []
        
        # Model performance recommendations
        if 'codebert' in self.cv_results and 'gnn' in self.cv_results:
            codebert_f1 = self.cv_results['codebert']['f1']['mean']
            gnn_f1 = self.cv_results['gnn']['f1_score']['mean']
            
            if codebert_f1 > gnn_f1:
                recommendations.append("CodeBERT shows better performance in cross-validation")
            else:
                recommendations.append("GNN shows better performance in cross-validation")
        
        # Statistical significance recommendations
        if 'statistical_analysis' in self.cv_results:
            for test_name, test_results in self.cv_results['statistical_analysis'].items():
                if test_results.get('significant', False):
                    recommendations.append(f"Statistically significant difference found in {test_name}")
        
        # General recommendations
        recommendations.extend([
            "Consider ensemble methods for improved performance",
            "Collect more data for underrepresented classes",
            "Fine-tune hyperparameters for better results",
            "Implement data augmentation techniques"
        ])
        
        return recommendations
    
    def _generate_cv_visualizations(self):
        """Generate cross-validation visualization plots"""
        try:
            # Model comparison plot
            self._plot_model_comparison()
            
            # Cross-validation scores plot
            self._plot_cv_scores()
            
            # Statistical significance plot
            if 'statistical_analysis' in self.cv_results:
                self._plot_statistical_analysis()
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
    
    def _plot_model_comparison(self):
        """Plot model performance comparison"""
        models = []
        f1_means = []
        f1_stds = []
        
        for model_name, model_results in self.cv_results.items():
            if model_name in ['codebert', 'gnn'] and model_results:
                f1_key = 'f1' if 'f1' in model_results else 'f1_score'
                if f1_key in model_results:
                    models.append(model_name.upper())
                    f1_means.append(model_results[f1_key]['mean'])
                    f1_stds.append(model_results[f1_key]['std'])
        
        if models:
            plt.figure(figsize=(10, 6))
            bars = plt.bar(models, f1_means, yerr=f1_stds, capsize=5, alpha=0.7,
                          color=['skyblue', 'lightcoral'])
            plt.ylabel('F1 Score')
            plt.title('Cross-Validation Performance Comparison')
            plt.ylim(0, 1)
            
            # Add value labels
            for bar, mean, std in zip(bars, f1_means, f1_stds):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                        f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / "cv_model_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_cv_scores(self):
        """Plot cross-validation scores distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for model_name, model_results in self.cv_results.items():
                if model_name in ['codebert', 'gnn'] and model_results:
                    metric_key = metric if metric in model_results else f'{metric}_score'
                    if metric_key in model_results:
                        scores = model_results[metric_key]['scores']
                        ax.hist(scores, alpha=0.7, label=model_name.upper(), bins=5)
            
            ax.set_xlabel(metric.capitalize())
            ax.set_ylabel('Frequency')
            ax.set_title(f'{metric.capitalize()} Score Distribution')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "cv_scores_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_statistical_analysis(self):
        """Plot statistical analysis results"""
        if not self.cv_results['statistical_analysis']:
            return
        
        tests = list(self.cv_results['statistical_analysis'].keys())
        p_values = [self.cv_results['statistical_analysis'][test]['p_value'] for test in tests]
        significant = [self.cv_results['statistical_analysis'][test]['significant'] for test in tests]
        
        plt.figure(figsize=(12, 6))
        colors = ['red' if sig else 'blue' for sig in significant]
        bars = plt.bar(tests, p_values, color=colors, alpha=0.7)
        
        # Add significance line
        plt.axhline(y=self.config['significance_level'], color='red', linestyle='--', 
                   label=f'Significance Level ({self.config["significance_level"]})')
        
        plt.ylabel('P-value')
        plt.title('Statistical Significance Tests')
        plt.xticks(rotation=45)
        plt.legend()
        
        # Add value labels
        for bar, p_val in zip(bars, p_values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{p_val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "statistical_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_markdown_report(self, summary_stats: Dict):
        """Generate markdown cross-validation report"""
        md_content = f"""# Cross-Validation Report

## Overview
- **Cross-Validation Date**: {datetime.now().isoformat()}
- **CV Folds**: {self.config['cv_folds']}
- **Random State**: {self.config['random_state']}

## Model Performance Summary

### CodeBERT
"""
        
        if 'codebert' in self.cv_results and self.cv_results['codebert']:
            codebert_results = self.cv_results['codebert']
            md_content += f"""
- **F1 Score**: {codebert_results['f1']['mean']:.4f} ± {codebert_results['f1']['std']:.4f}
- **Accuracy**: {codebert_results['accuracy']['mean']:.4f} ± {codebert_results['accuracy']['std']:.4f}
- **Precision**: {codebert_results['precision']['mean']:.4f} ± {codebert_results['precision']['std']:.4f}
- **Recall**: {codebert_results['recall']['mean']:.4f} ± {codebert_results['recall']['std']:.4f}
"""
        
        md_content += "\n### GNN\n"
        
        if 'gnn' in self.cv_results and self.cv_results['gnn']:
            gnn_results = self.cv_results['gnn']
            md_content += f"""
- **F1 Score**: {gnn_results['f1_score']['mean']:.4f} ± {gnn_results['f1_score']['std']:.4f}
- **Accuracy**: {gnn_results['accuracy']['mean']:.4f} ± {gnn_results['accuracy']['std']:.4f}
- **Precision**: {gnn_results['precision']['mean']:.4f} ± {gnn_results['precision']['std']:.4f}
- **Recall**: {gnn_results['recall']['mean']:.4f} ± {gnn_results['recall']['std']:.4f}
"""
        
        # Statistical analysis
        if 'statistical_analysis' in self.cv_results and self.cv_results['statistical_analysis']:
            md_content += "\n## Statistical Analysis\n"
            for test_name, test_results in self.cv_results['statistical_analysis'].items():
                md_content += f"""
### {test_name}
- **Test**: {test_results.get('test', 'N/A')}
- **P-value**: {test_results.get('p_value', 'N/A'):.4f}
- **Significant**: {test_results.get('significant', 'N/A')}
- **Effect Size**: {test_results.get('effect_size', 'N/A'):.4f}
"""
        
        # Recommendations
        md_content += "\n## Recommendations\n"
        for rec in summary_stats.get('recommendations', []):
            md_content += f"- {rec}\n"
        
        # Save report
        with open(self.output_dir / "cv_report.md", "w") as f:
            f.write(md_content)
    
    def run_full_cross_validation(self):
        """Run complete cross-validation pipeline"""
        logger.info("Starting full cross-validation pipeline...")
        
        # Load data
        df = self.load_benchmark_data()
        
        # Cross-validate models
        logger.info("Cross-validating CodeBERT model...")
        self.cross_validate_codebert(df)
        
        logger.info("Cross-validating GNN model...")
        self.cross_validate_gnn(df)
        
        # Run baseline cross-validation
        logger.info("Running baseline cross-validation...")
        self.run_baseline_cross_validation(df)
        
        # Perform statistical analysis
        logger.info("Performing statistical analysis...")
        self.perform_statistical_analysis()
        
        # Generate report
        logger.info("Generating cross-validation report...")
        summary_stats = self.generate_cv_report()
        
        logger.info("Cross-validation pipeline completed successfully!")
        logger.info(f"Results saved to {self.output_dir}")
        
        return self.cv_results

def main():
    """Main function for cross-validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Cross-validate vulnerability detection models')
    parser.add_argument('--data', required=True, help='Path to training data CSV')
    parser.add_argument('--output', default='benchmarks/cv_results', help='Output directory')
    parser.add_argument('--config', help='Path to cross-validation configuration JSON')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Initialize cross-validation framework
    framework = CrossValidationFramework(
        data_path=args.data,
        output_dir=args.output,
        config_path=args.config
    )
    
    # Update configuration
    framework.config['cv_folds'] = args.folds
    framework.config['random_state'] = args.random_state
    
    # Run cross-validation
    results = framework.run_full_cross_validation()
    
    print("Cross-validation completed successfully!")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
