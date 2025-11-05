#!/usr/bin/env python3
"""
Complete Pipeline Runner Script

This script runs the complete smart contract vulnerability detection pipeline:
1. Data collection from multiple sources
2. Data preprocessing and feature extraction
3. Dataset validation
4. Generation of analysis reports
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error:", e.stderr)
        return False

def main():
    """Main function to run the complete pipeline"""
    parser = argparse.ArgumentParser(description='Run complete smart contract vulnerability detection pipeline')
    parser.add_argument('--skip-collection', action='store_true',
                       help='Skip data collection step')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip data preprocessing step')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip validation step')
    parser.add_argument('--sources', nargs='+', 
                       choices=['smartbugs', 'github', 'etherscan', 'all'],
                       default=['all'],
                       help='Data sources to collect from')
    
    args = parser.parse_args()
    
    print("🚀 Starting Smart Contract Vulnerability Detection Pipeline")
    print(f"Sources: {args.sources}")
    print(f"Skip collection: {args.skip_collection}")
    print(f"Skip preprocessing: {args.skip_preprocessing}")
    print(f"Skip validation: {args.skip_validation}")
    
    # Step 1: Data Collection
    if not args.skip_collection:
        print("\n📊 Step 1: Data Collection")
        sources_str = ' '.join(args.sources)
        collection_cmd = f"python scripts/collect_data.py --sources {sources_str} --output data/raw"
        
        if not run_command(collection_cmd, "Collecting smart contract datasets"):
            print("❌ Data collection failed. Exiting.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping data collection")
    
    # Step 2: Data Preprocessing
    if not args.skip_preprocessing:
        print("\n🔧 Step 2: Data Preprocessing")
        preprocessing_cmd = "python scripts/preprocess_data.py --input data/raw --output data/processed"
        
        if not run_command(preprocessing_cmd, "Preprocessing collected data"):
            print("❌ Data preprocessing failed. Exiting.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping data preprocessing")
    
    # Step 3: Dataset Validation
    if not args.skip_validation:
        print("\n✅ Step 3: Dataset Validation")
        validation_cmd = "python tests/validate_dataset.py --dataset data/processed/combined_dataset.csv --output tests/validation_results"
        
        if not run_command(validation_cmd, "Validating dataset integrity and quality"):
            print("❌ Dataset validation failed. Exiting.")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping dataset validation")
    
    # Step 4: Generate Summary Report
    print("\n📋 Step 4: Generating Summary Report")
    
    # Check if files exist
    raw_files = list(Path("data/raw").glob("*.json"))
    processed_files = list(Path("data/processed").glob("*.csv"))
    validation_files = list(Path("tests/validation_results").glob("*.json"))
    
    summary = {
        "pipeline_status": "completed",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "raw_data_files": len(raw_files),
        "processed_data_files": len(processed_files),
        "validation_files": len(validation_files),
        "files_created": {
            "raw_data": [str(f) for f in raw_files],
            "processed_data": [str(f) for f in processed_files],
            "validation_results": [str(f) for f in validation_files]
        }
    }
    
    # Save summary
    import json
    with open("pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("✅ Pipeline completed successfully!")
    print(f"📁 Raw data files: {len(raw_files)}")
    print(f"📁 Processed data files: {len(processed_files)}")
    print(f"📁 Validation files: {len(validation_files)}")
    print(f"📄 Summary saved to: pipeline_summary.json")
    
    # Display next steps
    print("\n🎯 Next Steps:")
    print("1. Review the validation report in tests/validation_results/")
    print("2. Explore the data using notebooks/data_exploration.ipynb")
    print("3. Proceed with model development using the processed datasets")
    print("4. Check pipeline_summary.json for detailed results")

if __name__ == "__main__":
    main()
