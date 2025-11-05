#!/usr/bin/env python3
"""
Smart Contract Dataset Collection Script

This script collects smart contract data from various sources including:
- SmartBugs dataset
- GitHub repositories
- Etherscan verified contracts
"""

import os
import sys
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import yaml
from tqdm import tqdm
import time
import argparse

class SmartContractCollector:
    """Main class for collecting smart contract datasets"""
    
    def __init__(self, config_path: str = "config/vulnerability_categories.yaml"):
        """Initialize the collector with configuration"""
        self.config = self._load_config(config_path)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SmartContract-Vulnerability-Detection/1.0'
        })
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return {}
    
    def collect_smartbugs_data(self, output_dir: str) -> Dict:
        """Collect data from SmartBugs dataset"""
        print("Collecting SmartBugs dataset...")
        
        smartbugs_data = {
            'contracts': [],
            'vulnerabilities': [],
            'metadata': {
                'source': 'SmartBugs',
                'collection_date': pd.Timestamp.now().isoformat(),
                'total_contracts': 0,
                'total_vulnerabilities': 0
            }
        }
        
        # SmartBugs GitHub repository structure
        smartbugs_urls = [
            "https://raw.githubusercontent.com/smartbugs/smartbugs/master/dataset/reentrancy/",
            "https://raw.githubusercontent.com/smartbugs/smartbugs/master/dataset/integer_overflow/",
            "https://raw.githubusercontent.com/smartbugs/smartbugs/master/dataset/access_control/",
            "https://raw.githubusercontent.com/smartbugs/smartbugs/master/dataset/unchecked_external_calls/"
        ]
        
        for url in smartbugs_urls:
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    # Parse directory listing (simplified)
                    vulnerability_type = url.split('/')[-2]
                    self._process_smartbugs_directory(url, vulnerability_type, smartbugs_data)
            except Exception as e:
                print(f"Error collecting from {url}: {e}")
        
        # Save collected data
        output_path = Path(output_dir) / "smartbugs_data.json"
        with open(output_path, 'w') as f:
            json.dump(smartbugs_data, f, indent=2)
        
        print(f"SmartBugs data saved to {output_path}")
        return smartbugs_data
    
    def _process_smartbugs_directory(self, base_url: str, vuln_type: str, data: Dict):
        """Process SmartBugs directory and extract contract information"""
        # This is a simplified implementation
        # In practice, you would need to parse the actual directory structure
        
        # Mock data for demonstration
        mock_contracts = [
            {
                'name': f'{vuln_type}_contract_1.sol',
                'vulnerability_type': vuln_type,
                'severity': 'high',
                'source_url': f'{base_url}contract1.sol',
                'description': f'Example {vuln_type} vulnerability'
            },
            {
                'name': f'{vuln_type}_contract_2.sol',
                'vulnerability_type': vuln_type,
                'severity': 'medium',
                'source_url': f'{base_url}contract2.sol',
                'description': f'Another {vuln_type} vulnerability'
            }
        ]
        
        data['contracts'].extend(mock_contracts)
        data['vulnerabilities'].extend([
            {
                'type': vuln_type,
                'contract': contract['name'],
                'severity': contract['severity']
            } for contract in mock_contracts
        ])
    
    def collect_github_data(self, output_dir: str, search_terms: List[str] = None) -> Dict:
        """Collect data from GitHub repositories"""
        print("Collecting GitHub data...")
        
        if search_terms is None:
            search_terms = [
                "solidity smart contract vulnerability",
                "ethereum smart contract security",
                "defi smart contract audit"
            ]
        
        github_data = {
            'repositories': [],
            'contracts': [],
            'metadata': {
                'source': 'GitHub',
                'collection_date': pd.Timestamp.now().isoformat(),
                'search_terms': search_terms
            }
        }
        
        # GitHub API search for repositories
        for term in search_terms:
            try:
                # Note: In practice, you would use GitHub API with authentication
                # This is a simplified mock implementation
                mock_repos = [
                    {
                        'name': f'repo_{term.replace(" ", "_")}_1',
                        'url': f'https://github.com/user/repo_{term.replace(" ", "_")}_1',
                        'description': f'Smart contract repository for {term}',
                        'stars': 100,
                        'language': 'Solidity'
                    },
                    {
                        'name': f'repo_{term.replace(" ", "_")}_2',
                        'url': f'https://github.com/user/repo_{term.replace(" ", "_")}_2',
                        'description': f'Another repository for {term}',
                        'stars': 50,
                        'language': 'Solidity'
                    }
                ]
                
                github_data['repositories'].extend(mock_repos)
                
                # Add some delay to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"Error searching GitHub for '{term}': {e}")
        
        # Save GitHub data
        output_path = Path(output_dir) / "github_data.json"
        with open(output_path, 'w') as f:
            json.dump(github_data, f, indent=2)
        
        print(f"GitHub data saved to {output_path}")
        return github_data
    
    def collect_etherscan_data(self, output_dir: str, contract_addresses: List[str] = None) -> Dict:
        """Collect data from Etherscan verified contracts"""
        print("Collecting Etherscan data...")
        
        if contract_addresses is None:
            # Popular contract addresses for demonstration
            contract_addresses = [
                "0xA0b86a33E6441b8C4C8C0C4C8C0C4C8C0C4C8C0C4",  # Mock address
                "0xB1c97a44F6442b8C4C8C0C4C8C0C4C8C0C4C8C0C4",  # Mock address
            ]
        
        etherscan_data = {
            'contracts': [],
            'metadata': {
                'source': 'Etherscan',
                'collection_date': pd.Timestamp.now().isoformat(),
                'total_addresses': len(contract_addresses)
            }
        }
        
        for address in contract_addresses:
            try:
                # Mock contract data (in practice, you would use Etherscan API)
                contract_data = {
                    'address': address,
                    'name': f'Contract_{address[:8]}',
                    'source_code': '// Mock Solidity code\npragma solidity ^0.8.0;\ncontract MockContract {\n    // Contract implementation\n}',
                    'abi': '[]',
                    'compiler_version': 'v0.8.0',
                    'optimization': True
                }
                
                etherscan_data['contracts'].append(contract_data)
                
                # Add delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error collecting contract {address}: {e}")
        
        # Save Etherscan data
        output_path = Path(output_dir) / "etherscan_data.json"
        with open(output_path, 'w') as f:
            json.dump(etherscan_data, f, indent=2)
        
        print(f"Etherscan data saved to {output_path}")
        return etherscan_data
    
    def create_dataset_summary(self, output_dir: str) -> pd.DataFrame:
        """Create a summary of all collected datasets"""
        print("Creating dataset summary...")
        
        summary_data = []
        
        # Process each data source
        for source_file in Path(output_dir).glob("*.json"):
            try:
                with open(source_file, 'r') as f:
                    data = json.load(f)
                
                source_name = source_file.stem
                total_contracts = len(data.get('contracts', []))
                total_vulnerabilities = len(data.get('vulnerabilities', []))
                
                summary_data.append({
                    'source': source_name,
                    'total_contracts': total_contracts,
                    'total_vulnerabilities': total_vulnerabilities,
                    'file_path': str(source_file),
                    'collection_date': data.get('metadata', {}).get('collection_date', 'Unknown')
                })
                
            except Exception as e:
                print(f"Error processing {source_file}: {e}")
        
        # Create DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        summary_path = Path(output_dir) / "dataset_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Dataset summary saved to {summary_path}")
        print("\nDataset Summary:")
        print(summary_df.to_string(index=False))
        
        return summary_df

def main():
    """Main function to run data collection"""
    parser = argparse.ArgumentParser(description='Collect smart contract datasets')
    parser.add_argument('--output', '-o', default='data/raw', 
                       help='Output directory for collected data')
    parser.add_argument('--sources', nargs='+', 
                       choices=['smartbugs', 'github', 'etherscan', 'all'],
                       default=['all'],
                       help='Data sources to collect from')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = SmartContractCollector()
    
    print("Starting smart contract dataset collection...")
    print(f"Output directory: {args.output}")
    print(f"Sources: {args.sources}")
    
    # Collect data from specified sources
    if 'all' in args.sources or 'smartbugs' in args.sources:
        collector.collect_smartbugs_data(args.output)
    
    if 'all' in args.sources or 'github' in args.sources:
        collector.collect_github_data(args.output)
    
    if 'all' in args.sources or 'etherscan' in args.sources:
        collector.collect_etherscan_data(args.output)
    
    # Create summary
    collector.create_dataset_summary(args.output)
    
    print("\nData collection completed successfully!")

if __name__ == "__main__":
    main()
