#!/usr/bin/env python3
"""
Dataset Validation Script

This script validates the integrity and accuracy of the collected and preprocessed datasets:
- Checks data integrity and completeness
- Validates label accuracy
- Ensures reproducibility of preprocessing scripts
- Generates validation reports
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml
import hashlib
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class DatasetValidator:
    """Main class for validating smart contract datasets"""
    
    def __init__(self, config_path: str = "config/vulnerability_categories.yaml"):
        """Initialize the validator with configuration"""
        self.config = self._load_config(config_path)
        self.validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_passed': 0,
            'tests_failed': 0,
            'warnings': [],
            'errors': []
        }
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}
    
    def validate_data_integrity(self, dataset_path: str) -> Dict:
        """Validate data integrity and completeness"""
        print("Validating data integrity...")
        
        results = {
            'test_name': 'data_integrity',
            'passed': True,
            'issues': []
        }
        
        try:
            # Load dataset
            df = pd.read_csv(dataset_path)
            
            # Check basic structure
            if df.empty:
                results['issues'].append("Dataset is empty")
                results['passed'] = False
            
            # Check required columns
            required_columns = ['contract_name', 'source_code', 'vulnerability_type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                results['issues'].append(f"Missing required columns: {missing_columns}")
                results['passed'] = False
            
            # Check for null values in critical columns
            critical_columns = ['contract_name', 'source_code']
            for col in critical_columns:
                if col in df.columns:
                    null_count = df[col].isnull().sum()
                    if null_count > 0:
                        results['issues'].append(f"Column '{col}' has {null_count} null values")
                        results['passed'] = False
            
            # Check for duplicate contracts
            if 'contract_name' in df.columns:
                duplicate_count = df['contract_name'].duplicated().sum()
                if duplicate_count > 0:
                    results['issues'].append(f"Found {duplicate_count} duplicate contract names")
                    results['passed'] = False
            
            # Check source code quality
            if 'source_code' in df.columns:
                empty_code_count = (df['source_code'].str.len() == 0).sum()
                if empty_code_count > 0:
                    results['issues'].append(f"Found {empty_code_count} contracts with empty source code")
                    results['passed'] = False
                
                # Check for minimum code length
                min_code_length = 50  # Minimum characters for meaningful code
                short_code_count = (df['source_code'].str.len() < min_code_length).sum()
                if short_code_count > 0:
                    results['issues'].append(f"Found {short_code_count} contracts with very short source code (< {min_code_length} chars)")
            
            # Check data types
            if 'vulnerability_type' in df.columns:
                unique_types = df['vulnerability_type'].unique()
                print(f"Found vulnerability types: {unique_types}")
            
            if 'severity' in df.columns:
                unique_severities = df['severity'].unique()
                print(f"Found severity levels: {unique_severities}")
            
        except Exception as e:
            results['issues'].append(f"Error loading dataset: {str(e)}")
            results['passed'] = False
        
        return results
    
    def validate_label_accuracy(self, dataset_path: str) -> Dict:
        """Validate label accuracy and consistency"""
        print("Validating label accuracy...")
        
        results = {
            'test_name': 'label_accuracy',
            'passed': True,
            'issues': []
        }
        
        try:
            df = pd.read_csv(dataset_path)
            
            # Check vulnerability type labels
            if 'vulnerability_type' in df.columns:
                valid_vuln_types = list(self.config.get('vulnerability_categories', {}).keys())
                invalid_types = df[~df['vulnerability_type'].isin(valid_vuln_types + ['unknown'])]['vulnerability_type'].unique()
                
                if len(invalid_types) > 0:
                    results['issues'].append(f"Invalid vulnerability types found: {invalid_types}")
                    results['passed'] = False
                
                # Check for unknown labels
                unknown_count = (df['vulnerability_type'] == 'unknown').sum()
                if unknown_count > 0:
                    results['issues'].append(f"Found {unknown_count} contracts with unknown vulnerability type")
            
            # Check severity labels
            if 'severity' in df.columns:
                valid_severities = ['high', 'medium', 'low', 'critical', 'info', 'warning']
                invalid_severities = df[~df['severity'].isin(valid_severities + ['unknown'])]['severity'].unique()
                
                if len(invalid_severities) > 0:
                    results['issues'].append(f"Invalid severity levels found: {invalid_severities}")
                    results['passed'] = False
            
            # Check label distribution
            if 'vulnerability_type' in df.columns:
                label_distribution = df['vulnerability_type'].value_counts()
                print("Vulnerability type distribution:")
                print(label_distribution)
                
                # Check for class imbalance
                max_count = label_distribution.max()
                min_count = label_distribution.min()
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if imbalance_ratio > 10:  # Threshold for severe imbalance
                    results['issues'].append(f"Severe class imbalance detected (ratio: {imbalance_ratio:.2f})")
            
        except Exception as e:
            results['issues'].append(f"Error validating labels: {str(e)}")
            results['passed'] = False
        
        return results
    
    def validate_feature_quality(self, dataset_path: str) -> Dict:
        """Validate feature quality and consistency"""
        print("Validating feature quality...")
        
        results = {
            'test_name': 'feature_quality',
            'passed': True,
            'issues': []
        }
        
        try:
            df = pd.read_csv(dataset_path)
            
            # Check numeric features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            feature_columns = [col for col in numeric_columns if col not in ['vulnerability_type_encoded', 'severity_encoded']]
            
            for col in feature_columns:
                # Check for negative values in count features
                if 'count' in col.lower():
                    negative_count = (df[col] < 0).sum()
                    if negative_count > 0:
                        results['issues'].append(f"Column '{col}' has {negative_count} negative values")
                        results['passed'] = False
                
                # Check for extremely high values (potential outliers)
                if df[col].max() > 10000:  # Threshold for suspiciously high values
                    results['issues'].append(f"Column '{col}' has extremely high values (max: {df[col].max()})")
            
            # Check for missing values in features
            missing_values = df[feature_columns].isnull().sum()
            if missing_values.sum() > 0:
                results['issues'].append(f"Missing values found in features: {missing_values[missing_values > 0].to_dict()}")
                results['passed'] = False
            
            # Check feature correlation
            if len(feature_columns) > 1:
                correlation_matrix = df[feature_columns].corr()
                high_correlation_pairs = []
                
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        corr_value = correlation_matrix.iloc[i, j]
                        if abs(corr_value) > 0.95:  # High correlation threshold
                            high_correlation_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_value))
                
                if high_correlation_pairs:
                    results['issues'].append(f"High correlation found between features: {high_correlation_pairs}")
            
        except Exception as e:
            results['issues'].append(f"Error validating features: {str(e)}")
            results['passed'] = False
        
        return results
    
    def validate_reproducibility(self, dataset_path: str) -> Dict:
        """Validate reproducibility of preprocessing scripts"""
        print("Validating reproducibility...")
        
        results = {
            'test_name': 'reproducibility',
            'passed': True,
            'issues': []
        }
        
        try:
            df = pd.read_csv(dataset_path)
            
            # Check if dataset has consistent structure
            expected_columns = [
                'contract_name', 'vulnerability_type', 'severity', 'source_code',
                'total_lines', 'total_characters', 'function_count'
            ]
            
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                results['issues'].append(f"Missing expected columns: {missing_columns}")
                results['passed'] = False
            
            # Check data source consistency
            if 'data_source' in df.columns:
                source_distribution = df['data_source'].value_counts()
                print("Data source distribution:")
                print(source_distribution)
            
            # Check timestamp consistency
            if 'collection_date' in df.columns:
                unique_dates = df['collection_date'].unique()
                if len(unique_dates) > 1:
                    results['issues'].append(f"Multiple collection dates found: {unique_dates}")
            
        except Exception as e:
            results['issues'].append(f"Error validating reproducibility: {str(e)}")
            results['passed'] = False
        
        return results
    
    def generate_validation_report(self, results: List[Dict], output_path: str):
        """Generate comprehensive validation report"""
        print("Generating validation report...")
        
        report = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_tests': len(results),
            'passed_tests': sum(1 for r in results if r['passed']),
            'failed_tests': sum(1 for r in results if not r['passed']),
            'test_results': results,
            'summary': {
                'overall_status': 'PASSED' if all(r['passed'] for r in results) else 'FAILED',
                'critical_issues': [r['issues'] for r in results if not r['passed']],
                'warnings': [r['issues'] for r in results if r['passed'] and r['issues']]
            }
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"Validation report saved to {output_path}")
        
        # Print summary
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total tests: {report['total_tests']}")
        print(f"Passed: {report['passed_tests']}")
        print(f"Failed: {report['failed_tests']}")
        print(f"Overall status: {report['summary']['overall_status']}")
        
        if report['summary']['critical_issues']:
            print("\nCritical Issues:")
            for issues in report['summary']['critical_issues']:
                for issue in issues:
                    print(f"  - {issue}")
        
        if report['summary']['warnings']:
            print("\nWarnings:")
            for warnings in report['summary']['warnings']:
                for warning in warnings:
                    print(f"  - {warning}")
        
        return report
    
    def run_full_validation(self, dataset_path: str, output_dir: str) -> Dict:
        """Run complete validation suite"""
        print(f"Running full validation for dataset: {dataset_path}")
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Run all validation tests
        validation_results = []
        
        # Data integrity test
        integrity_result = self.validate_data_integrity(dataset_path)
        validation_results.append(integrity_result)
        
        # Label accuracy test
        label_result = self.validate_label_accuracy(dataset_path)
        validation_results.append(label_result)
        
        # Feature quality test
        feature_result = self.validate_feature_quality(dataset_path)
        validation_results.append(feature_result)
        
        # Reproducibility test
        reproducibility_result = self.validate_reproducibility(dataset_path)
        validation_results.append(reproducibility_result)
        
        # Generate report
        report_path = Path(output_dir) / "validation_report.json"
        report = self.generate_validation_report(validation_results, str(report_path))
        
        return report

def main():
    """Main function to run dataset validation"""
    parser = argparse.ArgumentParser(description='Validate smart contract datasets')
    parser.add_argument('--dataset', '-d', required=True,
                       help='Path to dataset CSV file')
    parser.add_argument('--output', '-o', default='tests/validation_results',
                       help='Output directory for validation results')
    parser.add_argument('--config', '-c', default='config/vulnerability_categories.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not Path(args.dataset).exists():
        print(f"Error: Dataset file {args.dataset} not found")
        sys.exit(1)
    
    # Initialize validator
    validator = DatasetValidator(args.config)
    
    print("Starting dataset validation...")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output}")
    
    # Run validation
    report = validator.run_full_validation(args.dataset, args.output)
    
    # Exit with appropriate code
    if report['summary']['overall_status'] == 'PASSED':
        print("\n✅ All validation tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some validation tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
