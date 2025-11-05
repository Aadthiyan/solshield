#!/usr/bin/env python3
"""
Unit Tests for Model Training and Evaluation Pipeline

This script contains comprehensive unit tests for the model training
and evaluation pipeline to ensure reproducibility and correctness.
"""

import unittest
import sys
import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import tempfile
import shutil

# Add project directories to path
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "training"))
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

from codebert_model import CodeBERTVulnerabilityDetector, CodeBERTTrainer, SmartContractDataset
from gnn_model import GNNVulnerabilityDetector, GNNTrainer, SmartContractGraphDataset, SolidityASTParser
from metrics import VulnerabilityDetectionMetrics
from model_comparison import ModelComparisonFramework

class TestCodeBERTModel(unittest.TestCase):
    """Test cases for CodeBERT model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_texts = [
            "contract TestContract { function test() public { require(msg.sender == owner); } }",
            "contract VulnerableContract { function withdraw() public { msg.sender.transfer(balance); } }",
            "contract SafeContract { function safeWithdraw() public { require(balance > 0); balance = 0; msg.sender.transfer(balance); } }"
        ]
        self.sample_labels = [0, 1, 0]  # 0: safe, 1: vulnerable
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_codebert_model_initialization(self):
        """Test CodeBERT model initialization"""
        model = CodeBERTVulnerabilityDetector(
            model_name="microsoft/codebert-base",
            num_classes=2,
            dropout_rate=0.1
        )
        
        self.assertIsInstance(model, CodeBERTVulnerabilityDetector)
        self.assertEqual(model.num_classes, 2)
        self.assertEqual(model.dropout_rate, 0.1)
    
    def test_codebert_forward_pass(self):
        """Test CodeBERT forward pass"""
        model = CodeBERTVulnerabilityDetector(num_classes=2)
        
        # Create dummy input
        batch_size = 2
        seq_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        labels = torch.tensor([0, 1])
        
        # Forward pass
        outputs = model(input_ids, attention_mask, labels)
        
        self.assertIn('loss', outputs)
        self.assertIn('logits', outputs)
        self.assertIsInstance(outputs['loss'], torch.Tensor)
        self.assertIsInstance(outputs['logits'], torch.Tensor)
        self.assertEqual(outputs['logits'].shape, (batch_size, 2))
    
    def test_smart_contract_dataset(self):
        """Test SmartContractDataset class"""
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        dataset = SmartContractDataset(
            texts=self.sample_texts,
            labels=self.sample_labels,
            tokenizer=tokenizer,
            max_length=128
        )
        
        self.assertEqual(len(dataset), 3)
        
        # Test dataset item
        item = dataset[0]
        self.assertIn('input_ids', item)
        self.assertIn('attention_mask', item)
        self.assertIn('labels', item)
        self.assertEqual(item['input_ids'].shape, (128,))
        self.assertEqual(item['attention_mask'].shape, (128,))
        self.assertIsInstance(item['labels'], torch.Tensor)
    
    def test_codebert_trainer_initialization(self):
        """Test CodeBERT trainer initialization"""
        trainer = CodeBERTTrainer(
            model_name="microsoft/codebert-base",
            num_classes=2,
            batch_size=8,
            learning_rate=1e-5
        )
        
        self.assertIsInstance(trainer, CodeBERTTrainer)
        self.assertEqual(trainer.batch_size, 8)
        self.assertEqual(trainer.learning_rate, 1e-5)
    
    def test_codebert_trainer_prepare_dataset(self):
        """Test dataset preparation"""
        trainer = CodeBERTTrainer()
        
        train_dataset, val_dataset = trainer.prepare_dataset(
            self.sample_texts, self.sample_labels, split_ratio=0.7
        )
        
        self.assertIsInstance(train_dataset, SmartContractDataset)
        self.assertIsInstance(val_dataset, SmartContractDataset)
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)
        self.assertEqual(len(train_dataset) + len(val_dataset), len(self.sample_texts))

class TestGNNModel(unittest.TestCase):
    """Test cases for GNN model"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_code = """
        contract TestContract {
            uint256 public balance;
            
            function deposit() public payable {
                balance += msg.value;
            }
            
            function withdraw(uint256 amount) public {
                require(amount <= balance, "Insufficient balance");
                balance -= amount;
                payable(msg.sender).transfer(amount);
            }
        }
        """
        
        self.sample_labels = [0, 1]  # 0: safe, 1: vulnerable
        
        # Create temporary directory for test outputs
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_solidity_ast_parser(self):
        """Test Solidity AST parser"""
        parser = SolidityASTParser()
        
        # Parse sample contract
        graph = parser.parse_contract(self.sample_code)
        
        self.assertIsNotNone(graph)
        self.assertGreater(len(graph.nodes()), 0)
        self.assertGreater(len(graph.edges()), 0)
    
    def test_gnn_model_initialization(self):
        """Test GNN model initialization"""
        model = GNNVulnerabilityDetector(
            input_dim=22,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            dropout=0.1,
            gnn_type='GCN'
        )
        
        self.assertIsInstance(model, GNNVulnerabilityDetector)
        self.assertEqual(model.input_dim, 22)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.output_dim, 2)
        self.assertEqual(model.num_layers, 2)
    
    def test_gnn_forward_pass(self):
        """Test GNN forward pass"""
        model = GNNVulnerabilityDetector(
            input_dim=22,
            hidden_dim=32,
            output_dim=2,
            num_layers=2
        )
        
        # Create dummy graph data
        batch_size = 2
        num_nodes = 10
        x = torch.randn(num_nodes, 22)
        edge_index = torch.randint(0, num_nodes, (2, 20))
        batch = torch.zeros(num_nodes, dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, batch=batch)
        
        # Forward pass
        logits = model(data)
        
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (batch_size, 2))
    
    def test_gnn_trainer_initialization(self):
        """Test GNN trainer initialization"""
        trainer = GNNTrainer(
            input_dim=22,
            hidden_dim=32,
            output_dim=2,
            num_layers=2,
            gnn_type='GCN',
            learning_rate=0.001
        )
        
        self.assertIsInstance(trainer, GNNTrainer)
        self.assertEqual(trainer.input_dim, 22)
        self.assertEqual(trainer.hidden_dim, 32)
        self.assertEqual(trainer.learning_rate, 0.001)
    
    def test_gnn_trainer_prepare_dataset(self):
        """Test GNN dataset preparation"""
        trainer = GNNTrainer(input_dim=22)
        
        source_codes = [self.sample_code, self.sample_code]
        labels = [0, 1]
        
        train_dataset, val_dataset = trainer.prepare_dataset(
            source_codes, labels, split_ratio=0.7
        )
        
        self.assertIsInstance(train_dataset, SmartContractGraphDataset)
        self.assertIsInstance(val_dataset, SmartContractGraphDataset)
        self.assertGreater(len(train_dataset), 0)
        self.assertGreater(len(val_dataset), 0)

class TestEvaluationMetrics(unittest.TestCase):
    """Test cases for evaluation metrics"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.metrics = VulnerabilityDetectionMetrics()
        
        # Create sample data
        self.y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
        self.y_pred = np.array([0, 1, 0, 0, 1, 0, 1, 1])
        self.y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4],
                               [0.2, 0.8], [0.7, 0.3], [0.1, 0.9], [0.4, 0.6]])
    
    def test_basic_metrics_computation(self):
        """Test basic metrics computation"""
        metrics = self.metrics.compute_basic_metrics(
            self.y_true, self.y_pred, self.y_prob
        )
        
        # Check that metrics are computed
        self.assertIn('accuracy', metrics)
        self.assertIn('precision_macro', metrics)
        self.assertIn('recall_macro', metrics)
        self.assertIn('f1_macro', metrics)
        
        # Check metric values are reasonable
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreaterEqual(metrics['f1_macro'], 0.0)
        self.assertLessEqual(metrics['f1_macro'], 1.0)
    
    def test_per_class_metrics(self):
        """Test per-class metrics computation"""
        per_class_metrics = self.metrics.compute_per_class_metrics(
            self.y_true, self.y_pred
        )
        
        # Check that per-class metrics are computed
        self.assertIn('class_0', per_class_metrics)
        self.assertIn('class_1', per_class_metrics)
        
        # Check metric structure
        for class_name, class_metrics in per_class_metrics.items():
            self.assertIn('precision', class_metrics)
            self.assertIn('recall', class_metrics)
            self.assertIn('f1_score', class_metrics)
            self.assertIn('support', class_metrics)
    
    def test_vulnerability_specific_metrics(self):
        """Test vulnerability-specific metrics"""
        vuln_metrics = self.metrics.compute_vulnerability_specific_metrics(
            self.y_true, self.y_pred
        )
        
        # Check that vulnerability metrics are computed
        self.assertIn('vulnerability_detection_rate', vuln_metrics)
        self.assertIn('false_positive_rate', vuln_metrics)
        self.assertIn('false_negative_rate', vuln_metrics)
        
        # Check metric values are reasonable
        for metric_name, metric_value in vuln_metrics.items():
            self.assertGreaterEqual(metric_value, 0.0)
            self.assertLessEqual(metric_value, 1.0)
    
    def test_efficiency_metrics(self):
        """Test efficiency metrics computation"""
        inference_times = [0.1, 0.2, 0.15, 0.18, 0.12]
        model_size = 100.5  # MB
        
        efficiency_metrics = self.metrics.compute_efficiency_metrics(
            inference_times, model_size
        )
        
        # Check that efficiency metrics are computed
        self.assertIn('avg_inference_time', efficiency_metrics)
        self.assertIn('throughput', efficiency_metrics)
        self.assertIn('model_size_mb', efficiency_metrics)
        
        # Check metric values are reasonable
        self.assertGreater(efficiency_metrics['avg_inference_time'], 0)
        self.assertGreater(efficiency_metrics['throughput'], 0)
        self.assertGreater(efficiency_metrics['model_size_mb'], 0)
    
    def test_model_evaluation(self):
        """Test comprehensive model evaluation"""
        evaluation_results = self.metrics.evaluate_model(
            self.y_true, self.y_pred, self.y_prob
        )
        
        # Check that all metric categories are present
        self.assertIn('basic_metrics', evaluation_results)
        self.assertIn('per_class_metrics', evaluation_results)
        self.assertIn('vulnerability_metrics', evaluation_results)
        self.assertIn('efficiency_metrics', evaluation_results)
        self.assertIn('classification_report', evaluation_results)
        self.assertIn('confusion_matrix', evaluation_results)
    
    def test_model_comparison(self):
        """Test model comparison functionality"""
        # Create mock model results
        model_results = {
            'model1': {
                'basic_metrics': {
                    'f1_weighted': 0.85,
                    'accuracy': 0.82,
                    'precision_weighted': 0.83,
                    'recall_weighted': 0.84
                }
            },
            'model2': {
                'basic_metrics': {
                    'f1_weighted': 0.78,
                    'accuracy': 0.75,
                    'precision_weighted': 0.76,
                    'recall_weighted': 0.77
                }
            }
        }
        
        comparison_results = self.metrics.compare_models(model_results)
        
        # Check that comparison results are generated
        self.assertIn('model_rankings', comparison_results)
        self.assertIn('best_model', comparison_results)
        self.assertIn('metric_comparisons', comparison_results)

class TestModelComparisonFramework(unittest.TestCase):
    """Test cases for model comparison framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'source_code': [
                "contract Test1 { function test() public { require(msg.sender == owner); } }",
                "contract Test2 { function withdraw() public { msg.sender.transfer(balance); } }",
                "contract Test3 { function safeWithdraw() public { require(balance > 0); balance = 0; msg.sender.transfer(balance); } }",
                "contract Test4 { function vulnerable() public { selfdestruct(msg.sender); } }"
            ],
            'vulnerability_type': ['safe', 'vulnerable', 'safe', 'vulnerable'],
            'vulnerability_type_encoded': [0, 1, 0, 1],
            'severity': ['low', 'high', 'low', 'high'],
            'data_source': ['test', 'test', 'test', 'test']
        })
        
        # Save sample data
        self.data_path = os.path.join(self.temp_dir, 'sample_data.csv')
        self.sample_data.to_csv(self.data_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        framework = ModelComparisonFramework(
            data_path=self.data_path,
            output_dir=os.path.join(self.temp_dir, 'output')
        )
        
        self.assertIsInstance(framework, ModelComparisonFramework)
        self.assertEqual(framework.data_path, self.data_path)
    
    def test_data_loading(self):
        """Test data loading functionality"""
        framework = ModelComparisonFramework(
            data_path=self.data_path,
            output_dir=os.path.join(self.temp_dir, 'output')
        )
        
        df = framework.load_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        self.assertIn('source_code', df.columns)
        self.assertIn('vulnerability_type_encoded', df.columns)
    
    def test_baseline_benchmarks(self):
        """Test baseline benchmark functionality"""
        framework = ModelComparisonFramework(
            data_path=self.data_path,
            output_dir=os.path.join(self.temp_dir, 'output')
        )
        
        df = framework.load_data()
        baseline_results = framework.run_baseline_benchmarks(df)
        
        # Check that baseline results are generated
        self.assertIn('random', baseline_results)
        self.assertIn('majority', baseline_results)
        
        # Check that metrics are computed
        for baseline_name, metrics in baseline_results.items():
            self.assertIn('accuracy', metrics)
            self.assertIn('f1', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)

class TestModelTrainingPipeline(unittest.TestCase):
    """Test cases for model training pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample training data
        self.sample_data = pd.DataFrame({
            'source_code': [
                "contract Test1 { function test() public { require(msg.sender == owner); } }",
                "contract Test2 { function withdraw() public { msg.sender.transfer(balance); } }",
                "contract Test3 { function safeWithdraw() public { require(balance > 0); balance = 0; msg.sender.transfer(balance); } }",
                "contract Test4 { function vulnerable() public { selfdestruct(msg.sender); } }",
                "contract Test5 { function safe() public { require(msg.value > 0); balance += msg.value; } }"
            ],
            'vulnerability_type': ['safe', 'vulnerable', 'safe', 'vulnerable', 'safe'],
            'vulnerability_type_encoded': [0, 1, 0, 1, 0],
            'severity': ['low', 'high', 'low', 'high', 'low'],
            'data_source': ['test', 'test', 'test', 'test', 'test']
        })
        
        # Save sample data
        self.data_path = os.path.join(self.temp_dir, 'training_data.csv')
        self.sample_data.to_csv(self.data_path, index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_preparation(self):
        """Test data preparation for training"""
        from training.train_models import ModelTrainingPipeline
        
        pipeline = ModelTrainingPipeline(
            data_path=self.data_path,
            output_dir=os.path.join(self.temp_dir, 'output')
        )
        
        df, data_info = pipeline.load_and_prepare_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(data_info, dict)
        self.assertEqual(len(df), 5)
        self.assertIn('total_samples', data_info)
        self.assertIn('vulnerability_distribution', data_info)

def run_model_tests():
    """Run all model-related tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestCodeBERTModel))
    test_suite.addTest(unittest.makeSuite(TestGNNModel))
    test_suite.addTest(unittest.makeSuite(TestEvaluationMetrics))
    test_suite.addTest(unittest.makeSuite(TestModelComparisonFramework))
    test_suite.addTest(unittest.makeSuite(TestModelTrainingPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()

def main():
    """Main function for running model tests"""
    print("Running model training and evaluation tests...")
    
    try:
        success = run_model_tests()
        
        if success:
            print("\n✅ All model tests passed!")
            return 0
        else:
            print("\n❌ Some model tests failed!")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
