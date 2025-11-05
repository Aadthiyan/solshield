#!/usr/bin/env python3
"""
Unit Tests for Data Preprocessing

This script contains unit tests for the data preprocessing functionality
to ensure reproducibility and correctness of preprocessing scripts.
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

from preprocess_data import SmartContractPreprocessor

class TestSmartContractPreprocessor(unittest.TestCase):
    """Test cases for SmartContractPreprocessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = SmartContractPreprocessor()
        
        # Create sample test data
        self.sample_code = """
        pragma solidity ^0.8.0;
        
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
        
        self.sample_contracts = [
            {
                'name': 'test_contract_1.sol',
                'vulnerability_type': 'reentrancy',
                'severity': 'high',
                'source_code': self.sample_code,
                'description': 'Test contract with reentrancy vulnerability'
            },
            {
                'name': 'test_contract_2.sol',
                'vulnerability_type': 'integer_overflow',
                'severity': 'medium',
                'source_code': 'contract Test { function add(uint a, uint b) public pure returns (uint) { return a + b; } }',
                'description': 'Test contract with integer overflow'
            }
        ]
    
    def test_clean_solidity_code(self):
        """Test Solidity code cleaning functionality"""
        # Test with sample code
        cleaned = self.preprocessor.clean_solidity_code(self.sample_code)
        
        # Check that comments are removed
        self.assertNotIn('//', cleaned)
        self.assertNotIn('/*', cleaned)
        
        # Check that pragma is removed
        self.assertNotIn('pragma solidity', cleaned)
        
        # Check that whitespace is normalized
        self.assertNotIn('\n\n', cleaned)
        
        # Check that code is not empty
        self.assertGreater(len(cleaned), 0)
    
    def test_clean_solidity_code_empty(self):
        """Test cleaning with empty code"""
        result = self.preprocessor.clean_solidity_code("")
        self.assertEqual(result, "")
        
        result = self.preprocessor.clean_solidity_code(None)
        self.assertEqual(result, "")
    
    def test_extract_features(self):
        """Test feature extraction from Solidity code"""
        features = self.preprocessor.extract_features(self.sample_code)
        
        # Check that features are extracted
        self.assertIsInstance(features, dict)
        self.assertGreater(len(features), 0)
        
        # Check specific features
        self.assertIn('total_lines', features)
        self.assertIn('total_characters', features)
        self.assertIn('function_count', features)
        self.assertIn('require_count', features)
        
        # Check that counts are non-negative
        for key, value in features.items():
            if 'count' in key:
                self.assertGreaterEqual(value, 0)
    
    def test_extract_features_empty(self):
        """Test feature extraction with empty code"""
        features = self.preprocessor.extract_features("")
        
        # Should return empty dict or dict with zeros
        self.assertIsInstance(features, dict)
    
    def test_map_vulnerability_types(self):
        """Test vulnerability type mapping"""
        # Test known mappings
        self.assertEqual(self.preprocessor.map_vulnerability_types('reentrancy'), 'reentrancy')
        self.assertEqual(self.preprocessor.map_vulnerability_types('integer_overflow'), 'integer_overflow')
        self.assertEqual(self.preprocessor.map_vulnerability_types('integer_underflow'), 'integer_overflow')
        
        # Test unknown mapping
        self.assertEqual(self.preprocessor.map_vulnerability_types('unknown_type'), 'unknown')
    
    def test_map_severity(self):
        """Test severity mapping"""
        # Test known mappings
        self.assertEqual(self.preprocessor.map_severity('high'), 'high')
        self.assertEqual(self.preprocessor.map_severity('medium'), 'medium')
        self.assertEqual(self.preprocessor.map_severity('critical'), 'high')
        
        # Test unknown mapping
        self.assertEqual(self.preprocessor.map_severity('unknown'), 'medium')
    
    def test_preprocess_contracts(self):
        """Test preprocessing of contract data"""
        # Create DataFrame from sample contracts
        df = pd.DataFrame(self.sample_contracts)
        
        # Test preprocessing
        processed_data = []
        for _, contract in df.iterrows():
            cleaned_code = self.preprocessor.clean_solidity_code(contract['source_code'])
            features = self.preprocessor.extract_features(cleaned_code)
            vuln_type = self.preprocessor.map_vulnerability_types(contract['vulnerability_type'])
            severity = self.preprocessor.map_severity(contract['severity'])
            
            processed_contract = {
                'contract_name': contract['name'],
                'vulnerability_type': vuln_type,
                'severity': severity,
                'source_code': cleaned_code,
                'original_code': contract['source_code'],
                'description': contract['description'],
                **features
            }
            processed_data.append(processed_contract)
        
        processed_df = pd.DataFrame(processed_data)
        
        # Check that data is processed correctly
        self.assertEqual(len(processed_df), 2)
        self.assertIn('contract_name', processed_df.columns)
        self.assertIn('vulnerability_type', processed_df.columns)
        self.assertIn('severity', processed_df.columns)
        self.assertIn('source_code', processed_df.columns)
        
        # Check that features are extracted
        feature_columns = [col for col in processed_df.columns if col.endswith('_count')]
        self.assertGreater(len(feature_columns), 0)
    
    def test_data_consistency(self):
        """Test data consistency after preprocessing"""
        df = pd.DataFrame(self.sample_contracts)
        
        # Process data
        processed_data = []
        for _, contract in df.iterrows():
            cleaned_code = self.preprocessor.clean_solidity_code(contract['source_code'])
            features = self.preprocessor.extract_features(cleaned_code)
            
            processed_contract = {
                'contract_name': contract['name'],
                'vulnerability_type': self.preprocessor.map_vulnerability_types(contract['vulnerability_type']),
                'severity': self.preprocessor.map_severity(contract['severity']),
                'source_code': cleaned_code,
                **features
            }
            processed_data.append(processed_contract)
        
        processed_df = pd.DataFrame(processed_data)
        
        # Check for missing values in critical columns
        critical_columns = ['contract_name', 'vulnerability_type', 'severity', 'source_code']
        for col in critical_columns:
            if col in processed_df.columns:
                null_count = processed_df[col].isnull().sum()
                self.assertEqual(null_count, 0, f"Column {col} has {null_count} null values")
        
        # Check that source code is not empty
        empty_code_count = (processed_df['source_code'].str.len() == 0).sum()
        self.assertEqual(empty_code_count, 0, f"Found {empty_code_count} contracts with empty source code")
    
    def test_feature_extraction_consistency(self):
        """Test that feature extraction is consistent"""
        # Extract features multiple times
        features1 = self.preprocessor.extract_features(self.sample_code)
        features2 = self.preprocessor.extract_features(self.sample_code)
        
        # Should be identical
        self.assertEqual(features1, features2)
        
        # Check that features are deterministic
        for key in features1:
            self.assertEqual(features1[key], features2[key])
    
    def test_vulnerability_type_consistency(self):
        """Test that vulnerability type mapping is consistent"""
        # Test multiple times with same input
        result1 = self.preprocessor.map_vulnerability_types('reentrancy')
        result2 = self.preprocessor.map_vulnerability_types('reentrancy')
        
        self.assertEqual(result1, result2)
        self.assertEqual(result1, 'reentrancy')
    
    def test_severity_mapping_consistency(self):
        """Test that severity mapping is consistent"""
        # Test multiple times with same input
        result1 = self.preprocessor.map_severity('high')
        result2 = self.preprocessor.map_severity('high')
        
        self.assertEqual(result1, result2)
        self.assertEqual(result1, 'high')

class TestDataQuality(unittest.TestCase):
    """Test cases for data quality validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessor = SmartContractPreprocessor()
        
        # Create sample dataset with quality issues
        self.quality_issues_data = pd.DataFrame({
            'contract_name': ['contract1', 'contract2', 'contract3', 'contract4'],
            'vulnerability_type': ['reentrancy', 'unknown', 'integer_overflow', 'reentrancy'],
            'severity': ['high', 'unknown', 'medium', 'high'],
            'source_code': ['contract code here', '', 'short', 'valid contract code with sufficient length'],
            'data_source': ['smartbugs', 'github', 'etherscan', 'smartbugs']
        })
    
    def test_identify_missing_values(self):
        """Test identification of missing values"""
        # Add some missing values
        test_data = self.quality_issues_data.copy()
        test_data.loc[1, 'vulnerability_type'] = None
        test_data.loc[2, 'severity'] = None
        
        # Check for missing values
        missing_values = test_data.isnull().sum()
        
        self.assertGreater(missing_values['vulnerability_type'], 0)
        self.assertGreater(missing_values['severity'], 0)
    
    def test_identify_empty_source_code(self):
        """Test identification of empty source code"""
        empty_code_count = (self.quality_issues_data['source_code'].str.len() == 0).sum()
        self.assertGreater(empty_code_count, 0)
    
    def test_identify_short_source_code(self):
        """Test identification of short source code"""
        short_code_count = (self.quality_issues_data['source_code'].str.len() < 50).sum()
        self.assertGreater(short_code_count, 0)
    
    def test_identify_unknown_labels(self):
        """Test identification of unknown labels"""
        unknown_vuln_count = (self.quality_issues_data['vulnerability_type'] == 'unknown').sum()
        unknown_severity_count = (self.quality_issues_data['severity'] == 'unknown').sum()
        
        self.assertGreater(unknown_vuln_count, 0)
        self.assertGreater(unknown_severity_count, 0)

def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSmartContractPreprocessor))
    test_suite.addTest(unittest.makeSuite(TestDataQuality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return success status
    return result.wasSuccessful()

if __name__ == "__main__":
    print("Running preprocessing unit tests...")
    success = run_tests()
    
    if success:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
