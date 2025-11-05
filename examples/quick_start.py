#!/usr/bin/env python3
"""
Quick Start Example

This script demonstrates how to use the smart contract vulnerability detection
dataset collection and preprocessing pipeline with minimal configuration.
"""

import os
import sys
import json
from pathlib import Path

# Add scripts to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from collect_data import SmartContractCollector
from preprocess_data import SmartContractPreprocessor
from validate_dataset import DatasetValidator

def quick_start_example():
    """Run a quick start example of the complete pipeline"""
    print("🚀 Smart Contract Vulnerability Detection - Quick Start")
    print("=" * 60)
    
    # Create necessary directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("tests/validation_results").mkdir(parents=True, exist_ok=True)
    
    # Step 1: Data Collection
    print("\n📊 Step 1: Collecting Sample Data")
    print("-" * 40)
    
    collector = SmartContractCollector()
    
    # Collect sample data from SmartBugs
    print("Collecting SmartBugs data...")
    smartbugs_data = collector.collect_smartbugs_data("data/raw")
    print(f"✅ Collected {len(smartbugs_data['contracts'])} contracts from SmartBugs")
    
    # Collect sample data from GitHub
    print("Collecting GitHub data...")
    github_data = collector.collect_github_data("data/raw")
    print(f"✅ Collected {len(github_data['repositories'])} repositories from GitHub")
    
    # Collect sample data from Etherscan
    print("Collecting Etherscan data...")
    etherscan_data = collector.collect_etherscan_data("data/raw")
    print(f"✅ Collected {len(etherscan_data['contracts'])} contracts from Etherscan")
    
    # Create dataset summary
    summary_df = collector.create_dataset_summary("data/raw")
    print(f"✅ Dataset summary created with {len(summary_df)} sources")
    
    # Step 2: Data Preprocessing
    print("\n🔧 Step 2: Preprocessing Data")
    print("-" * 40)
    
    preprocessor = SmartContractPreprocessor()
    
    # Create combined dataset
    print("Creating combined dataset...")
    combined_df = preprocessor.create_combined_dataset("data/raw", "data/processed/combined_dataset.csv")
    print(f"✅ Combined dataset created with {len(combined_df)} contracts")
    
    # Create train/test split
    print("Creating train/test split...")
    train_df, test_df = preprocessor.create_train_test_split(
        combined_df, "data/processed", test_size=0.2, random_state=42
    )
    print(f"✅ Training set: {len(train_df)} samples")
    print(f"✅ Test set: {len(test_df)} samples")
    
    # Step 3: Dataset Validation
    print("\n✅ Step 3: Validating Dataset")
    print("-" * 40)
    
    validator = DatasetValidator()
    
    # Run validation
    print("Running dataset validation...")
    validation_report = validator.run_full_validation(
        "data/processed/combined_dataset.csv", 
        "tests/validation_results"
    )
    
    # Check validation results
    if validation_report['summary']['overall_status'] == 'PASSED':
        print("✅ Dataset validation passed!")
    else:
        print("⚠️  Dataset validation found issues:")
        for issues in validation_report['summary']['critical_issues']:
            for issue in issues:
                print(f"   - {issue}")
    
    # Step 4: Generate Analysis Report
    print("\n📋 Step 4: Generating Analysis Report")
    print("-" * 40)
    
    # Dataset statistics
    stats = {
        'total_contracts': len(combined_df),
        'vulnerability_distribution': combined_df['vulnerability_type'].value_counts().to_dict(),
        'severity_distribution': combined_df['severity'].value_counts().to_dict(),
        'source_distribution': combined_df['data_source'].value_counts().to_dict(),
        'feature_columns': len([col for col in combined_df.columns if col.endswith('_count')])
    }
    
    # Save analysis report
    with open("data/processed/analysis_report.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    
    print("✅ Analysis report generated")
    
    # Step 5: Display Results
    print("\n📊 Step 5: Results Summary")
    print("-" * 40)
    
    print(f"📁 Total contracts processed: {len(combined_df)}")
    print(f"📁 Training samples: {len(train_df)}")
    print(f"📁 Test samples: {len(test_df)}")
    print(f"📁 Feature columns: {stats['feature_columns']}")
    print(f"📁 Validation status: {validation_report['summary']['overall_status']}")
    
    print("\n📈 Vulnerability Distribution:")
    for vuln_type, count in stats['vulnerability_distribution'].items():
        print(f"   {vuln_type}: {count}")
    
    print("\n📈 Severity Distribution:")
    for severity, count in stats['severity_distribution'].items():
        print(f"   {severity}: {count}")
    
    print("\n📈 Data Source Distribution:")
    for source, count in stats['source_distribution'].items():
        print(f"   {source}: {count}")
    
    # Step 6: Next Steps
    print("\n🎯 Next Steps")
    print("-" * 40)
    print("1. Review the validation report in tests/validation_results/")
    print("2. Explore the data using notebooks/data_exploration.ipynb")
    print("3. Check the analysis report in data/processed/analysis_report.json")
    print("4. Proceed with model development using the processed datasets")
    
    print("\n✅ Quick start example completed successfully!")
    
    return {
        'total_contracts': len(combined_df),
        'training_samples': len(train_df),
        'test_samples': len(test_df),
        'validation_status': validation_report['summary']['overall_status'],
        'files_created': [
            'data/raw/smartbugs_data.json',
            'data/raw/github_data.json',
            'data/raw/etherscan_data.json',
            'data/processed/combined_dataset.csv',
            'data/processed/train_dataset.csv',
            'data/processed/test_dataset.csv',
            'tests/validation_results/validation_report.json',
            'data/processed/analysis_report.json'
        ]
    }

if __name__ == "__main__":
    try:
        results = quick_start_example()
        print(f"\n🎉 Pipeline completed successfully!")
        print(f"📊 Results: {results}")
    except Exception as e:
        print(f"\n❌ Error during quick start: {e}")
        sys.exit(1)
