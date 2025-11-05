#!/usr/bin/env python3
"""
Smart Contract Data Preprocessing Script

This script preprocesses collected smart contract data for model training:
- Cleans and standardizes contract source code
- Maps vulnerabilities to standardized categories
- Creates training-ready datasets
- Generates feature vectors for ML models
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import argparse

class SmartContractPreprocessor:
    """Main class for preprocessing smart contract datasets"""
    
    def __init__(self, config_path: str = "config/vulnerability_categories.yaml"):
        """Initialize the preprocessor with configuration"""
        self.config = self._load_config(config_path)
        self.vulnerability_encoder = LabelEncoder()
        self.severity_encoder = LabelEncoder()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}
    
    def clean_solidity_code(self, code: str) -> str:
        """Clean and standardize Solidity code"""
        if not code or not isinstance(code, str):
            return ""
        
        # Remove comments
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        
        # Remove extra whitespace
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'\n\s*\n', '\n', code)
        
        # Remove pragma and import statements for standardization
        code = re.sub(r'pragma\s+solidity[^;]*;', '', code)
        code = re.sub(r'import\s+[^;]*;', '', code)
        
        # Remove contract/library/interface declarations for feature extraction
        code = re.sub(r'(contract|library|interface)\s+\w+', '', code)
        
        return code.strip()
    
    def extract_features(self, code: str) -> Dict[str, int]:
        """Extract features from Solidity code"""
        if not code:
            return {}
        
        features = {
            'total_lines': len(code.split('\n')),
            'total_characters': len(code),
            'function_count': len(re.findall(r'function\s+\w+', code)),
            'modifier_count': len(re.findall(r'modifier\s+\w+', code)),
            'event_count': len(re.findall(r'event\s+\w+', code)),
            'require_count': len(re.findall(r'require\s*\(', code)),
            'assert_count': len(re.findall(r'assert\s*\(', code)),
            'external_call_count': len(re.findall(r'\.call\s*\(', code)),
            'send_count': len(re.findall(r'\.send\s*\(', code)),
            'transfer_count': len(re.findall(r'\.transfer\s*\(', code)),
            'selfdestruct_count': len(re.findall(r'selfdestruct\s*\(', code)),
            'suicide_count': len(re.findall(r'suicide\s*\(', code)),
            'delegatecall_count': len(re.findall(r'delegatecall\s*\(', code)),
            'assembly_count': len(re.findall(r'assembly\s*{', code)),
            'tx_origin_count': len(re.findall(r'tx\.origin', code)),
            'block_timestamp_count': len(re.findall(r'block\.timestamp', code)),
            'block_number_count': len(re.findall(r'block\.number', code)),
            'msg_value_count': len(re.findall(r'msg\.value', code)),
            'msg_sender_count': len(re.findall(r'msg\.sender', code)),
            'now_count': len(re.findall(r'\bnow\b', code)),
            'gasleft_count': len(re.findall(r'gasleft\s*\(', code)),
            'keccak256_count': len(re.findall(r'keccak256\s*\(', code)),
            'sha3_count': len(re.findall(r'sha3\s*\(', code)),
            'ecrecover_count': len(re.findall(r'ecrecover\s*\(', code)),
            'revert_count': len(re.findall(r'revert\s*\(', code)),
            'throw_count': len(re.findall(r'throw\s*;', code)),
            'mapping_count': len(re.findall(r'mapping\s*\(', code)),
            'array_count': len(re.findall(r'\[\]', code)),
            'struct_count': len(re.findall(r'struct\s+\w+', code)),
            'enum_count': len(re.findall(r'enum\s+\w+', code)),
            'modifier_usage_count': len(re.findall(r'@\w+', code)),
            'payable_count': len(re.findall(r'payable', code)),
            'view_count': len(re.findall(r'view', code)),
            'pure_count': len(re.findall(r'pure', code)),
            'constant_count': len(re.findall(r'constant', code)),
            'immutable_count': len(re.findall(r'immutable', code)),
            'override_count': len(re.findall(r'override', code)),
            'virtual_count': len(re.findall(r'virtual', code)),
            'abstract_count': len(re.findall(r'abstract', code)),
            'interface_count': len(re.findall(r'interface', code)),
            'library_count': len(re.findall(r'library', code))
        }
        
        return features
    
    def map_vulnerability_types(self, vuln_type: str) -> str:
        """Map vulnerability types to standardized categories"""
        vuln_mapping = {
            'reentrancy': 'reentrancy',
            'integer_overflow': 'integer_overflow',
            'integer_underflow': 'integer_overflow',
            'access_control': 'access_control',
            'unchecked_external_calls': 'unchecked_external_calls',
            'front_running': 'front_running',
            'timestamp_dependence': 'timestamp_dependence',
            'gas_limit': 'gas_limit',
            'denial_of_service': 'denial_of_service',
            'tx_origin': 'tx_origin',
            'dos': 'denial_of_service',
            'unchecked_send': 'unchecked_external_calls',
            'unchecked_transfer': 'unchecked_external_calls',
            'reentrancy_eth': 'reentrancy',
            'reentrancy_no_eth': 'reentrancy'
        }
        
        return vuln_mapping.get(vuln_type.lower(), 'unknown')
    
    def map_severity(self, severity: str) -> str:
        """Map severity levels to standardized categories"""
        severity_mapping = {
            'high': 'high',
            'medium': 'medium',
            'low': 'low',
            'critical': 'high',
            'info': 'low',
            'warning': 'medium'
        }
        
        return severity_mapping.get(severity.lower(), 'medium')
    
    def preprocess_smartbugs_data(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Preprocess SmartBugs dataset"""
        print("Preprocessing SmartBugs data...")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        
        for contract in data.get('contracts', []):
            # Clean the source code
            cleaned_code = self.clean_solidity_code(contract.get('source_code', ''))
            
            # Extract features
            features = self.extract_features(cleaned_code)
            
            # Map vulnerability type
            vuln_type = self.map_vulnerability_types(contract.get('vulnerability_type', 'unknown'))
            
            # Map severity
            severity = self.map_severity(contract.get('severity', 'medium'))
            
            processed_contract = {
                'contract_name': contract.get('name', 'unknown'),
                'vulnerability_type': vuln_type,
                'severity': severity,
                'source_code': cleaned_code,
                'original_code': contract.get('source_code', ''),
                'description': contract.get('description', ''),
                'source_url': contract.get('source_url', ''),
                **features
            }
            
            processed_data.append(processed_contract)
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        print(f"Processed SmartBugs data saved to {output_path}")
        
        return df
    
    def preprocess_github_data(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Preprocess GitHub dataset"""
        print("Preprocessing GitHub data...")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        
        for repo in data.get('repositories', []):
            # Mock contract data (in practice, you would fetch actual contract files)
            mock_contract = {
                'contract_name': f"{repo['name']}_contract",
                'vulnerability_type': 'unknown',  # Would need manual labeling
                'severity': 'unknown',
                'source_code': '// Mock contract code\npragma solidity ^0.8.0;\ncontract MockContract {\n    function example() public {\n        // Implementation\n    }\n}',
                'original_code': '',
                'description': repo.get('description', ''),
                'source_url': repo.get('url', ''),
                'repository_name': repo['name'],
                'stars': repo.get('stars', 0),
                'language': repo.get('language', 'Solidity')
            }
            
            # Clean and extract features
            cleaned_code = self.clean_solidity_code(mock_contract['source_code'])
            features = self.extract_features(cleaned_code)
            
            processed_contract = {
                **mock_contract,
                'source_code': cleaned_code,
                **features
            }
            
            processed_data.append(processed_contract)
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        print(f"Processed GitHub data saved to {output_path}")
        
        return df
    
    def preprocess_etherscan_data(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Preprocess Etherscan dataset"""
        print("Preprocessing Etherscan data...")
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        processed_data = []
        
        for contract in data.get('contracts', []):
            # Clean the source code
            cleaned_code = self.clean_solidity_code(contract.get('source_code', ''))
            
            # Extract features
            features = self.extract_features(cleaned_code)
            
            processed_contract = {
                'contract_name': contract.get('name', 'unknown'),
                'vulnerability_type': 'unknown',  # Would need manual labeling
                'severity': 'unknown',
                'source_code': cleaned_code,
                'original_code': contract.get('source_code', ''),
                'description': f"Etherscan contract {contract.get('address', '')}",
                'source_url': f"https://etherscan.io/address/{contract.get('address', '')}",
                'contract_address': contract.get('address', ''),
                'compiler_version': contract.get('compiler_version', ''),
                'optimization': contract.get('optimization', False),
                **features
            }
            
            processed_data.append(processed_contract)
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        
        # Save processed data
        df.to_csv(output_path, index=False)
        print(f"Processed Etherscan data saved to {output_path}")
        
        return df
    
    def create_combined_dataset(self, input_dir: str, output_path: str) -> pd.DataFrame:
        """Create a combined dataset from all sources"""
        print("Creating combined dataset...")
        
        all_dataframes = []
        
        # Process each source
        for source_file in Path(input_dir).glob("*.json"):
            source_name = source_file.stem
            
            if source_name == "smartbugs_data":
                df = self.preprocess_smartbugs_data(str(source_file), 
                                                   str(Path(input_dir).parent / "processed" / f"{source_name}_processed.csv"))
            elif source_name == "github_data":
                df = self.preprocess_github_data(str(source_file), 
                                                str(Path(input_dir).parent / "processed" / f"{source_name}_processed.csv"))
            elif source_name == "etherscan_data":
                df = self.preprocess_etherscan_data(str(source_file), 
                                                  str(Path(input_dir).parent / "processed" / f"{source_name}_processed.csv"))
            else:
                continue
            
            # Add source column
            df['data_source'] = source_name
            all_dataframes.append(df)
        
        # Combine all datasets
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Encode categorical variables
        if 'vulnerability_type' in combined_df.columns:
            combined_df['vulnerability_type_encoded'] = self.vulnerability_encoder.fit_transform(
                combined_df['vulnerability_type'].fillna('unknown')
            )
        
        if 'severity' in combined_df.columns:
            combined_df['severity_encoded'] = self.severity_encoder.fit_transform(
                combined_df['severity'].fillna('unknown')
            )
        
        # Save combined dataset
        combined_df.to_csv(output_path, index=False)
        print(f"Combined dataset saved to {output_path}")
        
        # Create dataset statistics
        self._create_dataset_statistics(combined_df, output_path)
        
        return combined_df
    
    def _create_dataset_statistics(self, df: pd.DataFrame, output_path: str):
        """Create dataset statistics and save to file"""
        stats = {
            'total_contracts': len(df),
            'vulnerability_distribution': df['vulnerability_type'].value_counts().to_dict(),
            'severity_distribution': df['severity'].value_counts().to_dict(),
            'source_distribution': df['data_source'].value_counts().to_dict(),
            'feature_statistics': df.select_dtypes(include=[np.number]).describe().to_dict()
        }
        
        stats_path = Path(output_path).parent / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        print(f"Dataset statistics saved to {stats_path}")
    
    def create_train_test_split(self, df: pd.DataFrame, output_dir: str, 
                              test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/test split for the dataset"""
        print("Creating train/test split...")
        
        # Split the data
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, 
            stratify=df['vulnerability_type'] if 'vulnerability_type' in df.columns else None
        )
        
        # Save splits
        train_path = Path(output_dir) / "train_dataset.csv"
        test_path = Path(output_dir) / "test_dataset.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"Training set saved to {train_path} ({len(train_df)} samples)")
        print(f"Test set saved to {test_path} ({len(test_df)} samples)")
        
        return train_df, test_df

def main():
    """Main function to run data preprocessing"""
    parser = argparse.ArgumentParser(description='Preprocess smart contract datasets')
    parser.add_argument('--input', '-i', default='data/raw', 
                       help='Input directory containing raw data')
    parser.add_argument('--output', '-o', default='data/processed', 
                       help='Output directory for processed data')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size for train/test split')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = SmartContractPreprocessor()
    
    print("Starting smart contract data preprocessing...")
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    
    # Create combined dataset
    combined_df = preprocessor.create_combined_dataset(args.input, 
                                                      Path(args.output) / "combined_dataset.csv")
    
    # Create train/test split
    train_df, test_df = preprocessor.create_train_test_split(
        combined_df, args.output, args.test_size, args.random_state
    )
    
    print("\nData preprocessing completed successfully!")
    print(f"Total contracts processed: {len(combined_df)}")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")

if __name__ == "__main__":
    main()
