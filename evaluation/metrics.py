#!/usr/bin/env python3
"""
Evaluation Metrics for Smart Contract Vulnerability Detection

This module implements comprehensive evaluation metrics for assessing
model performance in vulnerability detection tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, average_precision_score
)
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VulnerabilityDetectionMetrics:
    """Comprehensive metrics for vulnerability detection evaluation"""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ['No Vulnerability', 'Vulnerability']
        self.metrics_history = []
    
    def compute_basic_metrics(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute basic classification metrics"""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Binary classification metrics (if applicable)
        if len(np.unique(y_true)) == 2:
            metrics['precision_binary'] = precision_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['recall_binary'] = recall_score(y_true, y_pred, average='binary', zero_division=0)
            metrics['f1_binary'] = f1_score(y_true, y_pred, average='binary', zero_division=0)
            
            # Additional binary metrics
            metrics['specificity'] = self._compute_specificity(y_true, y_pred)
            metrics['sensitivity'] = metrics['recall_binary']
            metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        
        # Advanced metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
        
        # Probability-based metrics
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1])
                    metrics['average_precision'] = average_precision_score(y_true, y_prob[:, 1])
                else:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
            except Exception as e:
                logger.warning(f"Could not compute probability-based metrics: {e}")
        
        return metrics
    
    def _compute_specificity(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute specificity (true negative rate)"""
        tn, fp, fn, tp = self._compute_confusion_matrix_values(y_true, y_pred)
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    def _compute_confusion_matrix_values(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
        """Compute confusion matrix values (TN, FP, FN, TP)"""
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # For multi-class, compute binary metrics for positive class
            tn = np.sum((y_true != 1) & (y_pred != 1))
            fp = np.sum((y_true != 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred != 1))
            tp = np.sum((y_true == 1) & (y_pred == 1))
        return tn, fp, fn, tp
    
    def compute_per_class_metrics(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute per-class metrics"""
        
        # Get unique classes
        classes = np.unique(np.concatenate([y_true, y_pred]))
        per_class_metrics = {}
        
        for cls in classes:
            # Binary classification for this class
            y_true_binary = (y_true == cls).astype(int)
            y_pred_binary = (y_pred == cls).astype(int)
            
            # Compute metrics
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Support (number of true instances)
            support = np.sum(y_true == cls)
            
            per_class_metrics[f'class_{cls}'] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'support': support
            }
        
        return per_class_metrics
    
    def compute_vulnerability_specific_metrics(self, 
                                             y_true: np.ndarray, 
                                             y_pred: np.ndarray,
                                             vulnerability_types: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute vulnerability-specific metrics"""
        
        metrics = {}
        
        # Vulnerability detection rate (recall for positive class)
        if len(np.unique(y_true)) == 2:
            metrics['vulnerability_detection_rate'] = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
            metrics['false_positive_rate'] = 1 - self._compute_specificity(y_true, y_pred)
            metrics['false_negative_rate'] = 1 - metrics['vulnerability_detection_rate']
        
        # Severity-weighted metrics (if vulnerability types provided)
        if vulnerability_types:
            # This would require severity mapping - simplified for now
            metrics['high_severity_detection_rate'] = self._compute_high_severity_detection(y_true, y_pred, vulnerability_types)
        
        return metrics
    
    def _compute_high_severity_detection(self, 
                                        y_true: np.ndarray, 
                                        y_pred: np.ndarray,
                                        vulnerability_types: List[str]) -> float:
        """Compute detection rate for high-severity vulnerabilities"""
        # Simplified implementation - in practice, would need severity mapping
        high_severity_types = ['reentrancy', 'integer_overflow', 'access_control']
        
        # Filter for high-severity vulnerabilities
        high_severity_mask = np.isin(vulnerability_types, high_severity_types)
        if np.sum(high_severity_mask) == 0:
            return 0.0
        
        y_true_high = y_true[high_severity_mask]
        y_pred_high = y_pred[high_severity_mask]
        
        return recall_score(y_true_high, y_pred_high, zero_division=0)
    
    def compute_efficiency_metrics(self, 
                                  inference_times: List[float],
                                  model_size: Optional[float] = None) -> Dict[str, float]:
        """Compute model efficiency metrics"""
        
        metrics = {}
        
        # Inference time metrics
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['std_inference_time'] = np.std(inference_times)
        metrics['min_inference_time'] = np.min(inference_times)
        metrics['max_inference_time'] = np.max(inference_times)
        metrics['median_inference_time'] = np.median(inference_times)
        
        # Throughput (samples per second)
        metrics['throughput'] = 1.0 / metrics['avg_inference_time'] if metrics['avg_inference_time'] > 0 else 0
        
        # Model size metrics
        if model_size is not None:
            metrics['model_size_mb'] = model_size
            metrics['model_size_gb'] = model_size / 1024
        
        return metrics
    
    def generate_classification_report(self, 
                                     y_true: np.ndarray, 
                                     y_pred: np.ndarray,
                                     target_names: Optional[List[str]] = None) -> str:
        """Generate detailed classification report"""
        
        return classification_report(
            y_true, y_pred, 
            target_names=target_names or self.class_names,
            zero_division=0
        )
    
    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            save_path: Optional[str] = None,
                            title: str = "Confusion Matrix") -> None:
        """Plot confusion matrix"""
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, 
                      y_true: np.ndarray, 
                      y_prob: np.ndarray,
                      save_path: Optional[str] = None,
                      title: str = "ROC Curve") -> None:
        """Plot ROC curve"""
        
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
            roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(title)
            plt.legend(loc="lower right")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def plot_precision_recall_curve(self, 
                                  y_true: np.ndarray, 
                                  y_prob: np.ndarray,
                                  save_path: Optional[str] = None,
                                  title: str = "Precision-Recall Curve") -> None:
        """Plot precision-recall curve"""
        
        if len(np.unique(y_true)) == 2:
            precision, recall, _ = precision_recall_curve(y_true, y_prob[:, 1])
            avg_precision = average_precision_score(y_true, y_prob[:, 1])
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision, color='darkorange', lw=2,
                    label=f'PR curve (AP = {avg_precision:.2f})')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(title)
            plt.legend(loc="lower left")
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
    
    def evaluate_model(self, 
                      y_true: np.ndarray, 
                      y_pred: np.ndarray,
                      y_prob: Optional[np.ndarray] = None,
                      vulnerability_types: Optional[List[str]] = None,
                      inference_times: Optional[List[float]] = None,
                      model_size: Optional[float] = None) -> Dict[str, Union[float, Dict]]:
        """Comprehensive model evaluation"""
        
        logger.info("Computing comprehensive evaluation metrics...")
        
        # Basic metrics
        basic_metrics = self.compute_basic_metrics(y_true, y_pred, y_prob)
        
        # Per-class metrics
        per_class_metrics = self.compute_per_class_metrics(y_true, y_pred)
        
        # Vulnerability-specific metrics
        vuln_metrics = self.compute_vulnerability_specific_metrics(y_true, y_pred, vulnerability_types)
        
        # Efficiency metrics
        efficiency_metrics = {}
        if inference_times:
            efficiency_metrics = self.compute_efficiency_metrics(inference_times, model_size)
        
        # Combine all metrics
        evaluation_results = {
            'basic_metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'vulnerability_metrics': vuln_metrics,
            'efficiency_metrics': efficiency_metrics,
            'classification_report': self.generate_classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Store in history
        self.metrics_history.append(evaluation_results)
        
        return evaluation_results
    
    def compare_models(self, 
                      model_results: Dict[str, Dict],
                      comparison_metrics: List[str] = None) -> Dict[str, Dict]:
        """Compare multiple models"""
        
        if comparison_metrics is None:
            comparison_metrics = ['f1_weighted', 'accuracy', 'precision_weighted', 'recall_weighted']
        
        comparison_results = {
            'model_rankings': {},
            'best_model': None,
            'metric_comparisons': {}
        }
        
        # Rank models by each metric
        for metric in comparison_metrics:
            model_scores = {}
            for model_name, results in model_results.items():
                if 'basic_metrics' in results and metric in results['basic_metrics']:
                    model_scores[model_name] = results['basic_metrics'][metric]
            
            if model_scores:
                # Sort by score (descending)
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                comparison_results['model_rankings'][metric] = sorted_models
                comparison_results['metric_comparisons'][metric] = model_scores
        
        # Determine best overall model (by F1 score)
        if 'f1_weighted' in comparison_results['metric_comparisons']:
            f1_scores = comparison_results['metric_comparisons']['f1_weighted']
            best_model = max(f1_scores.items(), key=lambda x: x[1])
            comparison_results['best_model'] = best_model[0]
        
        return comparison_results
    
    def save_evaluation_results(self, 
                               results: Dict, 
                               save_path: str) -> None:
        """Save evaluation results to file"""
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {save_path}")
    
    def generate_evaluation_report(self, 
                                 results: Dict,
                                 save_path: str) -> None:
        """Generate comprehensive evaluation report"""
        
        report_content = f"""# Model Evaluation Report

## Overview
- **Evaluation Date**: {pd.Timestamp.now().isoformat()}
- **Total Samples**: {len(results.get('confusion_matrix', []))}

## Basic Metrics
- **Accuracy**: {results['basic_metrics'].get('accuracy', 'N/A'):.4f}
- **F1 Score (Weighted)**: {results['basic_metrics'].get('f1_weighted', 'N/A'):.4f}
- **Precision (Weighted)**: {results['basic_metrics'].get('precision_weighted', 'N/A'):.4f}
- **Recall (Weighted)**: {results['basic_metrics'].get('recall_weighted', 'N/A'):.4f}

## Advanced Metrics
- **Cohen's Kappa**: {results['basic_metrics'].get('cohen_kappa', 'N/A'):.4f}
- **Matthews Correlation Coefficient**: {results['basic_metrics'].get('matthews_corrcoef', 'N/A'):.4f}

## Classification Report
```
{results['classification_report']}
```

## Per-Class Metrics
"""
        
        for class_name, metrics in results['per_class_metrics'].items():
            report_content += f"""
### {class_name}
- **Precision**: {metrics['precision']:.4f}
- **Recall**: {metrics['recall']:.4f}
- **F1 Score**: {metrics['f1_score']:.4f}
- **Support**: {metrics['support']}
"""
        
        if results['efficiency_metrics']:
            report_content += f"""
## Efficiency Metrics
- **Average Inference Time**: {results['efficiency_metrics'].get('avg_inference_time', 'N/A'):.4f}s
- **Throughput**: {results['efficiency_metrics'].get('throughput', 'N/A'):.2f} samples/sec
- **Model Size**: {results['efficiency_metrics'].get('model_size_mb', 'N/A'):.2f} MB
"""
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Evaluation report saved to {save_path}")

def main():
    """Example usage of evaluation metrics"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model performance')
    parser.add_argument('--predictions', required=True, help='Path to predictions CSV')
    parser.add_argument('--output', default='evaluation/results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load predictions
    df = pd.read_csv(args.predictions)
    y_true = df['true_labels'].values
    y_pred = df['predicted_labels'].values
    y_prob = df[['prob_class_0', 'prob_class_1']].values if 'prob_class_0' in df.columns else None
    
    # Initialize metrics
    metrics = VulnerabilityDetectionMetrics()
    
    # Evaluate model
    results = metrics.evaluate_model(y_true, y_pred, y_prob)
    
    # Save results
    metrics.save_evaluation_results(results, f"{args.output}/evaluation_results.json")
    metrics.generate_evaluation_report(results, f"{args.output}/evaluation_report.md")
    
    print("Evaluation completed!")

if __name__ == "__main__":
    main()
