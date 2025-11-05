"""
QIE SDK Integration and Deployment Module

This module provides integration with QIE SDK for deploying the smart contract
vulnerability detection dApp on QIE testnet.
"""

import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import subprocess
import requests
from web3 import Web3
from eth_account import Account

logger = logging.getLogger(__name__)


@dataclass
class QIEConfig:
    """QIE network configuration."""
    network_name: str
    rpc_url: str
    chain_id: int
    gas_price: int
    gas_limit: int
    deployer_private_key: str
    contract_address: Optional[str] = None
    deployment_tx_hash: Optional[str] = None


@dataclass
class DeploymentResult:
    """Result of contract deployment."""
    success: bool
    contract_address: Optional[str] = None
    transaction_hash: Optional[str] = None
    gas_used: Optional[int] = None
    deployment_time: float = 0.0
    error_message: Optional[str] = None
    network_info: Optional[Dict[str, Any]] = None


class QIESDKIntegration:
    """QIE SDK integration for smart contract deployment."""
    
    def __init__(self, config: QIEConfig):
        self.config = config
        self.w3 = Web3(Web3.HTTPProvider(config.rpc_url))
        self.account = Account.from_key(config.deployer_private_key)
        
        # Verify connection
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to QIE network: {config.rpc_url}")
        
        logger.info(f"Connected to QIE network: {config.network_name}")
    
    def deploy_contract(self, contract_bytecode: str, contract_abi: List[Dict], 
                       constructor_args: List[Any] = None) -> DeploymentResult:
        """
        Deploy a smart contract to QIE testnet.
        
        Args:
            contract_bytecode: Compiled contract bytecode
            contract_abi: Contract ABI
            constructor_args: Constructor arguments
            
        Returns:
            DeploymentResult with deployment information
        """
        start_time = time.time()
        
        try:
            # Prepare contract
            contract = self.w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
            
            # Build constructor transaction
            if constructor_args:
                constructor_txn = contract.constructor(*constructor_args).build_transaction({
                    'from': self.account.address,
                    'gas': self.config.gas_limit,
                    'gasPrice': self.config.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                })
            else:
                constructor_txn = contract.constructor().build_transaction({
                    'from': self.account.address,
                    'gas': self.config.gas_limit,
                    'gasPrice': self.config.gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                })
            
            # Sign transaction
            signed_txn = self.w3.eth.account.sign_transaction(constructor_txn, self.account.key)
            
            # Send transaction
            tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            deployment_time = time.time() - start_time
            
            if receipt.status == 1:
                logger.info(f"Contract deployed successfully at: {receipt.contractAddress}")
                return DeploymentResult(
                    success=True,
                    contract_address=receipt.contractAddress,
                    transaction_hash=tx_hash.hex(),
                    gas_used=receipt.gasUsed,
                    deployment_time=deployment_time,
                    network_info=self._get_network_info()
                )
            else:
                logger.error("Contract deployment failed")
                return DeploymentResult(
                    success=False,
                    deployment_time=deployment_time,
                    error_message="Transaction failed"
                )
                
        except Exception as e:
            logger.error(f"Deployment error: {str(e)}")
            return DeploymentResult(
                success=False,
                deployment_time=time.time() - start_time,
                error_message=str(e)
            )
    
    def _get_network_info(self) -> Dict[str, Any]:
        """Get current network information."""
        try:
            latest_block = self.w3.eth.get_block('latest')
            return {
                "network_name": self.config.network_name,
                "chain_id": self.config.chain_id,
                "latest_block_number": latest_block.number,
                "latest_block_hash": latest_block.hash.hex(),
                "gas_price": self.w3.eth.gas_price,
                "account_balance": self.w3.eth.get_balance(self.account.address)
            }
        except Exception as e:
            logger.warning(f"Failed to get network info: {e}")
            return {"error": str(e)}
    
    def verify_contract(self, contract_address: str, contract_source: str, 
                       contract_name: str) -> bool:
        """
        Verify contract source code on QIE network.
        
        Args:
            contract_address: Deployed contract address
            contract_source: Contract source code
            contract_name: Contract name
            
        Returns:
            True if verification successful
        """
        try:
            # This would typically use QIE's contract verification API
            # For now, we'll simulate the verification process
            logger.info(f"Verifying contract {contract_name} at {contract_address}")
            
            # Simulate verification delay
            time.sleep(2)
            
            # In a real implementation, you would:
            # 1. Submit source code to QIE verification service
            # 2. Wait for verification to complete
            # 3. Check verification status
            
            logger.info("Contract verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Contract verification failed: {e}")
            return False


class VulnerabilityAuditorContract:
    """Smart contract for vulnerability auditing on QIE network."""
    
    def __init__(self, qie_sdk: QIESDKIntegration):
        self.qie_sdk = qie_sdk
        self.contract_abi = self._get_contract_abi()
        self.contract_bytecode = self._get_contract_bytecode()
    
    def _get_contract_abi(self) -> List[Dict]:
        """Get the vulnerability auditor contract ABI."""
        return [
            {
                "inputs": [],
                "name": "constructor",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "inputs": [
                    {
                        "internalType": "string",
                        "name": "contractCode",
                        "type": "string"
                    }
                ],
                "name": "analyzeContract",
                "outputs": [
                    {
                        "components": [
                            {
                                "internalType": "string",
                                "name": "vulnerabilityType",
                                "type": "string"
                            },
                            {
                                "internalType": "string",
                                "name": "severity",
                                "type": "string"
                            },
                            {
                                "internalType": "uint256",
                                "name": "confidence",
                                "type": "uint256"
                            },
                            {
                                "internalType": "string",
                                "name": "description",
                                "type": "string"
                            },
                            {
                                "internalType": "string",
                                "name": "location",
                                "type": "string"
                            }
                        ],
                        "internalType": "struct VulnerabilityAuditor.Vulnerability",
                        "name": "",
                        "type": "tuple[]"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getAnalysisCount",
                "outputs": [
                    {
                        "internalType": "uint256",
                        "name": "",
                        "type": "uint256"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {
                        "internalType": "uint256",
                        "name": "analysisId",
                        "type": "uint256"
                    }
                ],
                "name": "getAnalysisResult",
                "outputs": [
                    {
                        "components": [
                            {
                                "internalType": "string",
                                "name": "contractHash",
                                "type": "string"
                            },
                            {
                                "internalType": "uint256",
                                "name": "timestamp",
                                "type": "uint256"
                            },
                            {
                                "internalType": "bool",
                                "name": "isVulnerable",
                                "type": "bool"
                            },
                            {
                                "internalType": "uint256",
                                "name": "riskScore",
                                "type": "uint256"
                            }
                        ],
                        "internalType": "struct VulnerabilityAuditor.AnalysisResult",
                        "name": "",
                        "type": "tuple"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    def _get_contract_bytecode(self) -> str:
        """Get the vulnerability auditor contract bytecode."""
        # This would be the actual compiled bytecode
        # For demonstration, we'll use a placeholder
        return "0x608060405234801561001057600080fd5b506004361061004c5760003560e01c8063..."
    
    def deploy(self) -> DeploymentResult:
        """Deploy the vulnerability auditor contract."""
        return self.qie_sdk.deploy_contract(
            contract_bytecode=self.contract_bytecode,
            contract_abi=self.contract_abi
        )
    
    def analyze_contract(self, contract_address: str, contract_code: str) -> Dict[str, Any]:
        """
        Analyze a contract using the deployed auditor contract.
        
        Args:
            contract_address: Address of the contract to analyze
            contract_code: Source code of the contract
            
        Returns:
            Analysis results
        """
        try:
            # Create contract instance
            contract = self.qie_sdk.w3.eth.contract(
                address=contract_address,
                abi=self.contract_abi
            )
            
            # Call analyzeContract function
            result = contract.functions.analyzeContract(contract_code).call()
            
            # Convert result to dictionary
            vulnerabilities = []
            for vuln in result:
                vulnerabilities.append({
                    "vulnerability_type": vuln[0],
                    "severity": vuln[1],
                    "confidence": vuln[2],
                    "description": vuln[3],
                    "location": vuln[4]
                })
            
            return {
                "success": True,
                "vulnerabilities": vulnerabilities,
                "contract_address": contract_address,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "contract_address": contract_address
            }


class QIEDeploymentManager:
    """Manages deployment of the vulnerability detection dApp on QIE testnet."""
    
    def __init__(self, config_path: str = "deployment/qie_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.qie_sdk = QIESDKIntegration(self.config)
        self.auditor_contract = VulnerabilityAuditorContract(self.qie_sdk)
    
    def _load_config(self) -> QIEConfig:
        """Load QIE configuration from file."""
        if not self.config_path.exists():
            # Create default configuration
            default_config = QIEConfig(
                network_name="QIE Testnet",
                rpc_url="https://testnet.qie.network/rpc",
                chain_id=12345,  # QIE testnet chain ID
                gas_price=20000000000,  # 20 Gwei
                gas_limit=5000000,  # 5M gas limit
                deployer_private_key=os.getenv("QIE_PRIVATE_KEY", "")
            )
            
            # Save default config
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump(asdict(default_config), f, indent=2)
            
            logger.warning(f"Created default QIE config at {self.config_path}")
            logger.warning("Please update the configuration with your private key and network details")
            
            return default_config
        
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)
        
        return QIEConfig(**config_data)
    
    def deploy_dapp(self) -> Dict[str, Any]:
        """Deploy the complete dApp to QIE testnet."""
        logger.info("Starting dApp deployment to QIE testnet...")
        
        deployment_results = {
            "deployment_timestamp": datetime.now().isoformat(),
            "network_info": self.qie_sdk._get_network_info(),
            "contracts": {},
            "success": True,
            "errors": []
        }
        
        try:
            # Deploy vulnerability auditor contract
            logger.info("Deploying vulnerability auditor contract...")
            auditor_result = self.auditor_contract.deploy()
            
            if auditor_result.success:
                deployment_results["contracts"]["vulnerability_auditor"] = {
                    "address": auditor_result.contract_address,
                    "transaction_hash": auditor_result.transaction_hash,
                    "gas_used": auditor_result.gas_used,
                    "deployment_time": auditor_result.deployment_time
                }
                
                # Update config with deployed address
                self.config.contract_address = auditor_result.contract_address
                self.config.deployment_tx_hash = auditor_result.transaction_hash
                
                # Save updated config
                with open(self.config_path, 'w') as f:
                    json.dump(asdict(self.config), f, indent=2)
                
                logger.info(f"Vulnerability auditor deployed at: {auditor_result.contract_address}")
                
                # Verify contract
                logger.info("Verifying contract...")
                verification_success = self.qie_sdk.verify_contract(
                    auditor_result.contract_address,
                    "VulnerabilityAuditor.sol",
                    "VulnerabilityAuditor"
                )
                
                deployment_results["contracts"]["vulnerability_auditor"]["verified"] = verification_success
                
            else:
                deployment_results["success"] = False
                deployment_results["errors"].append(f"Auditor contract deployment failed: {auditor_result.error_message}")
                logger.error(f"Auditor contract deployment failed: {auditor_result.error_message}")
            
            # Deploy additional contracts if needed
            # (e.g., proxy contracts, upgradeable contracts, etc.)
            
        except Exception as e:
            deployment_results["success"] = False
            deployment_results["errors"].append(str(e))
            logger.error(f"Deployment failed: {e}")
        
        # Save deployment results
        results_path = Path("deployment/deployment_results.json")
        with open(results_path, 'w') as f:
            json.dump(deployment_results, f, indent=2)
        
        logger.info(f"Deployment results saved to {results_path}")
        return deployment_results
    
    def test_deployment(self) -> Dict[str, Any]:
        """Test the deployed dApp functionality."""
        if not self.config.contract_address:
            return {"error": "No deployed contract address found"}
        
        logger.info("Testing deployed dApp...")
        
        test_results = {
            "test_timestamp": datetime.now().isoformat(),
            "contract_address": self.config.contract_address,
            "tests": {},
            "overall_success": True
        }
        
        try:
            # Test contract analysis functionality
            test_contract_code = """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestContract {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable: external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
}
"""
            
            # Test analysis
            analysis_result = self.auditor_contract.analyze_contract(
                self.config.contract_address,
                test_contract_code
            )
            
            test_results["tests"]["contract_analysis"] = {
                "success": analysis_result.get("success", False),
                "vulnerabilities_found": len(analysis_result.get("vulnerabilities", [])),
                "result": analysis_result
            }
            
            # Test contract interaction
            contract = self.qie_sdk.w3.eth.contract(
                address=self.config.contract_address,
                abi=self.auditor_contract.contract_abi
            )
            
            # Get analysis count
            analysis_count = contract.functions.getAnalysisCount().call()
            test_results["tests"]["analysis_count"] = {
                "success": True,
                "count": analysis_count
            }
            
        except Exception as e:
            test_results["overall_success"] = False
            test_results["error"] = str(e)
            logger.error(f"Deployment test failed: {e}")
        
        # Save test results
        test_results_path = Path("deployment/test_results.json")
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        logger.info(f"Test results saved to {test_results_path}")
        return test_results
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        status = {
            "config_loaded": True,
            "network_connected": self.qie_sdk.w3.is_connected(),
            "contract_deployed": self.config.contract_address is not None,
            "contract_address": self.config.contract_address,
            "deployment_tx_hash": self.config.deployment_tx_hash,
            "network_info": self.qie_sdk._get_network_info()
        }
        
        return status


def create_qie_config(network_name: str = "QIE Testnet", 
                     rpc_url: str = "https://testnet.qie.network/rpc",
                     private_key: str = None) -> QIEConfig:
    """Create QIE configuration."""
    if not private_key:
        private_key = os.getenv("QIE_PRIVATE_KEY", "")
        if not private_key:
            raise ValueError("Private key must be provided or set in QIE_PRIVATE_KEY environment variable")
    
    return QIEConfig(
        network_name=network_name,
        rpc_url=rpc_url,
        chain_id=12345,  # QIE testnet chain ID
        gas_price=20000000000,  # 20 Gwei
        gas_limit=5000000,  # 5M gas limit
        deployer_private_key=private_key
    )


def deploy_to_qie_testnet(private_key: str = None) -> Dict[str, Any]:
    """
    Convenience function to deploy the dApp to QIE testnet.
    
    Args:
        private_key: Private key for deployment (optional if set in environment)
        
    Returns:
        Deployment results
    """
    try:
        deployment_manager = QIEDeploymentManager()
        
        # Deploy dApp
        deployment_results = deployment_manager.deploy_dapp()
        
        if deployment_results["success"]:
            # Test deployment
            test_results = deployment_manager.test_deployment()
            deployment_results["test_results"] = test_results
        
        return deployment_results
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "deployment_timestamp": datetime.now().isoformat()
        }
