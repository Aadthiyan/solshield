#!/usr/bin/env python3
"""
Complete Model Training and Evaluation Example

This script demonstrates the complete pipeline for training and evaluating
smart contract vulnerability detection models using CodeBERT and GNN.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project directories to path
sys.path.append(str(Path(__file__).parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent / "training"))
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))
sys.path.append(str(Path(__file__).parent.parent / "benchmarks"))

from train_models import ModelTrainingPipeline
from model_comparison import ModelComparisonFramework
from cross_validation import CrossValidationFramework
from metrics import VulnerabilityDetectionMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_data(output_path: str, num_samples: int = 100) -> str:
    """Create sample data for demonstration"""
    logger.info(f"Creating sample data with {num_samples} samples...")
    
    # Generate sample smart contract code
    sample_contracts = [
        # Safe contracts
        "contract SafeContract { function deposit() public payable { balance += msg.value; } }",
        "contract SecureContract { function withdraw(uint amount) public { require(amount <= balance); balance -= amount; msg.sender.transfer(amount); } }",
        "contract ProtectedContract { modifier onlyOwner() { require(msg.sender == owner); _; } function adminFunction() public onlyOwner { } }",
        
        # Vulnerable contracts
        "contract VulnerableContract { function withdraw() public { msg.sender.transfer(balance); balance = 0; } }",
        "contract ReentrancyVulnerable { function withdraw() public { msg.sender.call{value: balance}(''); balance = 0; } }",
        "contract IntegerOverflow { function add(uint a, uint b) public pure returns (uint) { return a + b; } }",
        "contract AccessControl { function adminFunction() public { selfdestruct(msg.sender); } }",
        "contract UncheckedCall { function externalCall() public { msg.sender.call{value: 1 ether}(''); } }"
    ]
    
    # Generate sample data
    data = []
    for i in range(num_samples):
        # Randomly select contract template
        contract_template = np.random.choice(sample_contracts)
        
        # Add some variation
        if "Safe" in contract_template or "Secure" in contract_template or "Protected" in contract_template:
            vulnerability_type = "safe"
            severity = "low"
            encoded_type = 0
        else:
            vulnerability_type = "vulnerable"
            severity = "high"
            encoded_type = 1
        
        # Add some noise to make it more realistic
        if np.random.random() < 0.1:  # 10% chance of mislabeling
            vulnerability_type = "safe" if vulnerability_type == "vulnerable" else "vulnerable"
            encoded_type = 1 - encoded_type
        
        data.append({
            'contract_name': f'contract_{i}',
            'source_code': contract_template,
            'vulnerability_type': vulnerability_type,
            'vulnerability_type_encoded': encoded_type,
            'severity': severity,
            'data_source': 'sample',
            'description': f'Sample contract {i}',
            'total_lines': len(contract_template.split('\n')),
            'total_characters': len(contract_template),
            'function_count': contract_template.count('function'),
            'require_count': contract_template.count('require'),
            'assert_count': contract_template.count('assert'),
            'external_call_count': contract_template.count('.call'),
            'transfer_count': contract_template.count('.transfer'),
            'send_count': contract_template.count('.send'),
            'selfdestruct_count': contract_template.count('selfdestruct'),
            'delegatecall_count': contract_template.count('delegatecall'),
            'assembly_count': contract_template.count('assembly'),
            'tx_origin_count': contract_template.count('tx.origin'),
            'block_timestamp_count': contract_template.count('block.timestamp'),
            'msg_value_count': contract_template.count('msg.value'),
            'msg_sender_count': contract_template.count('msg.sender'),
            'now_count': contract_template.count('now'),
            'gasleft_count': contract_template.count('gasleft'),
            'keccak256_count': contract_template.count('keccak256'),
            'sha3_count': contract_template.count('sha3'),
            'ecrecover_count': contract_template.count('ecrecover'),
            'revert_count': contract_template.count('revert'),
            'throw_count': contract_template.count('throw'),
            'mapping_count': contract_template.count('mapping'),
            'array_count': contract_template.count('[]'),
            'struct_count': contract_template.count('struct'),
            'enum_count': contract_template.count('enum'),
            'modifier_usage_count': contract_template.count('@'),
            'payable_count': contract_template.count('payable'),
            'view_count': contract_template.count('view'),
            'pure_count': contract_template.count('pure'),
            'constant_count': contract_template.count('constant'),
            'immutable_count': contract_template.count('immutable'),
            'override_count': contract_template.count('override'),
            'virtual_count': contract_template.count('virtual'),
            'abstract_count': contract_template.count('abstract'),
            'interface_count': contract_template.count('interface'),
            'library_count': contract_template.count('library')
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data created with {len(df)} samples")
    logger.info(f"Vulnerability distribution: {df['vulnerability_type'].value_counts().to_dict()}")
    
    return output_path

def run_model_training(data_path: str, output_dir: str) -> dict:
    """Run model training pipeline"""
    logger.info("Starting model training pipeline...")
    
    # Initialize training pipeline
    pipeline = ModelTrainingPipeline(
        data_path=data_path,
        output_dir=output_dir
    )
    
    # Run full pipeline
    results = pipeline.run_full_pipeline()
    
    logger.info("Model training completed successfully!")
    
    return results

def run_model_comparison(data_path: str, output_dir: str) -> dict:
    """Run model comparison framework"""
    logger.info("Starting model comparison...")
    
    # Initialize comparison framework
    framework = ModelComparisonFramework(
        data_path=data_path,
        output_dir=output_dir
    )
    
    # Run comparison
    results = framework.run_full_comparison()
    
    logger.info("Model comparison completed successfully!")
    
    return results

def run_cross_validation(data_path: str, output_dir: str) -> dict:
    """Run cross-validation framework"""
    logger.info("Starting cross-validation...")
    
    # Initialize cross-validation framework
    framework = CrossValidationFramework(
        data_path=data_path,
        output_dir=output_dir
    )
    
    # Run cross-validation
    results = framework.run_full_cross_validation()
    
    logger.info("Cross-validation completed successfully!")
    
    return results

def generate_final_report(training_results: dict, comparison_results: dict, cv_results: dict, output_dir: str):
    """Generate comprehensive final report"""
    logger.info("Generating final report...")
    
    # Create reports directory
    reports_dir = Path(output_dir) / "final_reports"
    reports_dir.mkdir(exist_ok=True)
    
    # Generate comprehensive report
    final_report = {
        'execution_summary': {
            'timestamp': datetime.now().isoformat(),
            'total_models_trained': 2,
            'cross_validation_folds': cv_results.get('metadata', {}).get('cv_folds', 'N/A'),
            'total_samples': len(pd.read_csv(training_results.get('metadata', {}).get('data_path', ''))) if 'data_path' in training_results.get('metadata', {}) else 'N/A'
        },
        'model_performance': {
            'codebert': training_results.get('codebert', {}),
            'gnn': training_results.get('gnn', {})
        },
        'model_comparison': comparison_results.get('model_comparison', {}),
        'cross_validation_results': {
            'codebert_cv': cv_results.get('codebert', {}),
            'gnn_cv': cv_results.get('gnn', {})
        },
        'statistical_analysis': {
            'comparison_tests': comparison_results.get('statistical_tests', {}),
            'cv_statistical_tests': cv_results.get('statistical_analysis', {})
        },
        'recommendations': {
            'best_model': comparison_results.get('model_comparison', {}).get('best_model', 'N/A'),
            'performance_insights': [],
            'next_steps': []
        }
    }
    
    # Generate recommendations
    if 'codebert' in training_results and 'gnn' in training_results:
        codebert_f1 = training_results['codebert'].get('validation_metrics', {}).get('f1', 0)
        gnn_f1 = training_results['gnn'].get('validation_metrics', {}).get('f1_score', 0)
        
        if codebert_f1 > gnn_f1:
            final_report['recommendations']['best_model'] = 'CodeBERT'
            final_report['recommendations']['performance_insights'].append("CodeBERT shows better performance for this dataset")
        else:
            final_report['recommendations']['best_model'] = 'GNN'
            final_report['recommendations']['performance_insights'].append("GNN shows better performance for this dataset")
    
    # Add general recommendations
    final_report['recommendations']['next_steps'].extend([
        "Consider ensemble methods for improved performance",
        "Collect more data for underrepresented vulnerability types",
        "Fine-tune hyperparameters for better results",
        "Implement data augmentation techniques",
        "Deploy the best performing model for production use"
    ])
    
    # Save final report
    with open(reports_dir / "final_report.json", "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    
    # Generate markdown report
    md_content = f"""# Smart Contract Vulnerability Detection - Final Report

## Executive Summary
- **Execution Date**: {final_report['execution_summary']['timestamp']}
- **Total Models Trained**: {final_report['execution_summary']['total_models_trained']}
- **Cross-Validation Folds**: {final_report['execution_summary']['cross_validation_folds']}
- **Total Samples**: {final_report['execution_summary']['total_samples']}

## Model Performance Summary

### CodeBERT Model
- **Validation F1**: {training_results.get('codebert', {}).get('validation_metrics', {}).get('f1', 'N/A')}
- **Validation Accuracy**: {training_results.get('codebert', {}).get('validation_metrics', {}).get('accuracy', 'N/A')}

### GNN Model
- **Validation F1**: {training_results.get('gnn', {}).get('validation_metrics', {}).get('f1_score', 'N/A')}
- **Validation Accuracy**: {training_results.get('gnn', {}).get('validation_metrics', {}).get('accuracy', 'N/A')}

## Cross-Validation Results

### CodeBERT Cross-Validation
- **CV F1 Mean**: {cv_results.get('codebert', {}).get('f1', {}).get('mean', 'N/A')} ± {cv_results.get('codebert', {}).get('f1', {}).get('std', 'N/A')}
- **CV Accuracy Mean**: {cv_results.get('codebert', {}).get('accuracy', {}).get('mean', 'N/A')} ± {cv_results.get('codebert', {}).get('accuracy', {}).get('std', 'N/A')}

### GNN Cross-Validation
- **CV F1 Mean**: {cv_results.get('gnn', {}).get('f1_score', {}).get('mean', 'N/A')} ± {cv_results.get('gnn', {}).get('f1_score', {}).get('std', 'N/A')}
- **CV Accuracy Mean**: {cv_results.get('gnn', {}).get('accuracy', {}).get('mean', 'N/A')} ± {cv_results.get('gnn', {}).get('accuracy', {}).get('std', 'N/A')}

## Recommendations

### Best Model
**{final_report['recommendations']['best_model']}** shows the best performance for this dataset.

### Performance Insights
{chr(10).join(f"- {insight}" for insight in final_report['recommendations']['performance_insights'])}

### Next Steps
{chr(10).join(f"- {step}" for step in final_report['recommendations']['next_steps'])}

## Files Generated
- `training_summary.json`: Complete training results
- `comparison_report.json`: Model comparison results
- `cv_results.json`: Cross-validation results
- `final_report.json`: Comprehensive final report
- Various visualization plots and detailed reports

## Conclusion
The model training and evaluation pipeline has been successfully executed. Both CodeBERT and GNN models have been trained, evaluated, and compared using comprehensive metrics and cross-validation. The results provide insights into model performance and recommendations for production deployment.
"""
    
    with open(reports_dir / "final_report.md", "w") as f:
        f.write(md_content)
    
    logger.info(f"Final report generated in {reports_dir}")
    
    return final_report

def main():
    """Main function for complete model training and evaluation"""
    parser = argparse.ArgumentParser(description='Complete model training and evaluation pipeline')
    parser.add_argument('--data', help='Path to training data CSV (if not provided, sample data will be created)')
    parser.add_argument('--output', default='examples/output', help='Output directory')
    parser.add_argument('--samples', type=int, default=100, help='Number of sample data points to create')
    parser.add_argument('--skip-training', action='store_true', help='Skip model training')
    parser.add_argument('--skip-comparison', action='store_true', help='Skip model comparison')
    parser.add_argument('--skip-cv', action='store_true', help='Skip cross-validation')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Create sample data if not provided
    if not args.data:
        logger.info("No data provided, creating sample data...")
        data_path = create_sample_data(f"{args.output}/sample_data.csv", args.samples)
    else:
        data_path = args.data
        logger.info(f"Using provided data: {data_path}")
    
    # Initialize results storage
    training_results = {}
    comparison_results = {}
    cv_results = {}
    
    try:
        # Run model training
        if not args.skip_training:
            logger.info("=" * 60)
            logger.info("PHASE 1: MODEL TRAINING")
            logger.info("=" * 60)
            training_results = run_model_training(data_path, f"{args.output}/training")
        
        # Run model comparison
        if not args.skip_comparison:
            logger.info("=" * 60)
            logger.info("PHASE 2: MODEL COMPARISON")
            logger.info("=" * 60)
            comparison_results = run_model_comparison(data_path, f"{args.output}/comparison")
        
        # Run cross-validation
        if not args.skip_cv:
            logger.info("=" * 60)
            logger.info("PHASE 3: CROSS-VALIDATION")
            logger.info("=" * 60)
            cv_results = run_cross_validation(data_path, f"{args.output}/cross_validation")
        
        # Generate final report
        logger.info("=" * 60)
        logger.info("PHASE 4: FINAL REPORT GENERATION")
        logger.info("=" * 60)
        final_report = generate_final_report(training_results, comparison_results, cv_results, args.output)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Model training: {'Completed' if training_results else 'Skipped'}")
        logger.info(f"✅ Model comparison: {'Completed' if comparison_results else 'Skipped'}")
        logger.info(f"✅ Cross-validation: {'Completed' if cv_results else 'Skipped'}")
        logger.info(f"✅ Final report: Generated")
        logger.info(f"📁 Results saved to: {args.output}")
        
        if final_report['recommendations']['best_model'] != 'N/A':
            logger.info(f"🏆 Best model: {final_report['recommendations']['best_model']}")
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
