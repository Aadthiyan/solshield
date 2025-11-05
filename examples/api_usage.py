#!/usr/bin/env python3
"""
API Usage Examples

This script demonstrates how to use the Smart Contract Vulnerability Detection API
with various examples and use cases.
"""

import requests
import json
import time
from typing import Dict, List, Any
import asyncio
import aiohttp

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "analyze": f"{API_BASE_URL}/api/v1/analyze",
    "report": f"{API_BASE_URL}/api/v1/report",
    "batch": f"{API_BASE_URL}/api/v1/analyze/batch",
    "health": f"{API_BASE_URL}/api/v1/health",
    "status": f"{API_BASE_URL}/api/v1/status",
    "metrics": f"{API_BASE_URL}/api/v1/metrics"
}

class VulnerabilityDetectionClient:
    """Client for interacting with the vulnerability detection API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        return response.json()
    
    def analyze_contract(self, 
                        contract_code: str, 
                        contract_name: str = None,
                        model_type: str = "ensemble",
                        include_optimization: bool = True,
                        include_explanation: bool = True) -> Dict[str, Any]:
        """Analyze a smart contract for vulnerabilities"""
        
        request_data = {
            "contract_code": contract_code,
            "model_type": model_type,
            "include_optimization_suggestions": include_optimization,
            "include_explanation": include_explanation
        }
        
        if contract_name:
            request_data["contract_name"] = contract_name
        
        response = self.session.post(
            f"{self.base_url}/api/v1/analyze",
            json=request_data
        )
        
        return response.json()
    
    def get_report(self, report_id: str) -> Dict[str, Any]:
        """Get vulnerability report by ID"""
        response = self.session.get(f"{self.base_url}/api/v1/report/{report_id}")
        return response.json()
    
    def batch_analyze(self, contracts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze multiple contracts in batch"""
        request_data = {
            "contracts": contracts
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/analyze/batch",
            json=request_data
        )
        
        return response.json()
    
    def get_batch_reports(self, batch_id: str) -> Dict[str, Any]:
        """Get batch analysis reports"""
        response = self.session.get(f"{self.base_url}/api/v1/batch/{batch_id}")
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        response = self.session.get(f"{self.base_url}/api/v1/status")
        return response.json()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        response = self.session.get(f"{self.base_url}/api/v1/metrics")
        return response.json()

def example_safe_contract():
    """Example: Analyze a safe contract"""
    print("=== Safe Contract Analysis ===")
    
    client = VulnerabilityDetectionClient()
    
    # Safe contract example
    safe_contract = """
    contract SafeContract {
        uint256 public balance;
        address public owner;
        
        modifier onlyOwner() {
            require(msg.sender == owner, "Not the owner");
            _;
        }
        
        constructor() {
            owner = msg.sender;
        }
        
        function deposit() public payable {
            balance += msg.value;
        }
        
        function withdraw(uint256 amount) public onlyOwner {
            require(amount <= balance, "Insufficient balance");
            balance -= amount;
            payable(owner).transfer(amount);
        }
    }
    """
    
    # Analyze the contract
    result = client.analyze_contract(
        contract_code=safe_contract,
        contract_name="SafeContract",
        model_type="ensemble"
    )
    
    print(f"Analysis Result: {result}")
    
    if result.get("success"):
        report_id = result["report_id"]
        print(f"Report ID: {report_id}")
        
        # Get detailed report
        report = client.get_report(report_id)
        print(f"Report: {json.dumps(report, indent=2)}")
    
    return result

def example_vulnerable_contract():
    """Example: Analyze a vulnerable contract"""
    print("\n=== Vulnerable Contract Analysis ===")
    
    client = VulnerabilityDetectionClient()
    
    # Vulnerable contract example (Reentrancy)
    vulnerable_contract = """
    contract VulnerableContract {
        mapping(address => uint256) public balances;
        
        function deposit() public payable {
            balances[msg.sender] += msg.value;
        }
        
        function withdraw() public {
            uint256 amount = balances[msg.sender];
            require(amount > 0, "No balance");
            balances[msg.sender] = 0;
            msg.sender.transfer(amount);
        }
    }
    """
    
    # Analyze the contract
    result = client.analyze_contract(
        contract_code=vulnerable_contract,
        contract_name="VulnerableContract",
        model_type="ensemble"
    )
    
    print(f"Analysis Result: {result}")
    
    if result.get("success"):
        report_id = result["report_id"]
        print(f"Report ID: {report_id}")
        
        # Get detailed report
        report = client.get_report(report_id)
        if report.get("success"):
            report_data = report["report"]
            print(f"Vulnerable: {report_data['is_vulnerable']}")
            print(f"Risk Score: {report_data['risk_score']}")
            print(f"Vulnerabilities: {len(report_data['vulnerabilities'])}")
            
            for vuln in report_data['vulnerabilities']:
                print(f"  - {vuln['type']}: {vuln['description']}")
    
    return result

def example_batch_analysis():
    """Example: Batch analysis of multiple contracts"""
    print("\n=== Batch Analysis Example ===")
    
    client = VulnerabilityDetectionClient()
    
    # Multiple contracts to analyze
    contracts = [
        {
            "contract_code": """
            contract Batch1 {
                function test() public {
                    require(msg.sender != address(0));
                }
            }
            """,
            "model_type": "ensemble"
        },
        {
            "contract_code": """
            contract Batch2 {
                function vulnerable() public {
                    msg.sender.transfer(address(this).balance);
                }
            }
            """,
            "model_type": "codebert"
        },
        {
            "contract_code": """
            contract Batch3 {
                function complex() public {
                    uint256 a = 1;
                    uint256 b = 2;
                    uint256 c = a + b;
                }
            }
            """,
            "model_type": "gnn"
        }
    ]
    
    # Submit batch analysis
    batch_result = client.batch_analyze(contracts)
    print(f"Batch Result: {batch_result}")
    
    if batch_result.get("success"):
        batch_id = batch_result["batch_id"]
        print(f"Batch ID: {batch_id}")
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Get batch reports
        batch_reports = client.get_batch_reports(batch_id)
        print(f"Batch Reports: {json.dumps(batch_reports, indent=2)}")
    
    return batch_result

def example_model_comparison():
    """Example: Compare different models"""
    print("\n=== Model Comparison Example ===")
    
    client = VulnerabilityDetectionClient()
    
    # Test contract
    test_contract = """
    contract ModelTest {
        function test() public {
            require(msg.sender != address(0));
            msg.sender.transfer(1 ether);
        }
    }
    """
    
    models = ["codebert", "gnn", "ensemble"]
    results = {}
    
    for model in models:
        print(f"\nTesting {model} model...")
        
        result = client.analyze_contract(
            contract_code=test_contract,
            contract_name=f"ModelTest_{model}",
            model_type=model
        )
        
        if result.get("success"):
            report_id = result["report_id"]
            report = client.get_report(report_id)
            
            if report.get("success"):
                report_data = report["report"]
                results[model] = {
                    "vulnerable": report_data["is_vulnerable"],
                    "confidence": report_data["overall_confidence"],
                    "risk_score": report_data["risk_score"],
                    "vulnerability_count": len(report_data["vulnerabilities"])
                }
        
        time.sleep(1)  # Rate limiting
    
    print("\nModel Comparison Results:")
    for model, result in results.items():
        print(f"{model}: Vulnerable={result['vulnerable']}, "
              f"Confidence={result['confidence']:.2f}, "
              f"Risk={result['risk_score']:.1f}, "
              f"Vulns={result['vulnerability_count']}")
    
    return results

def example_system_monitoring():
    """Example: System monitoring and metrics"""
    print("\n=== System Monitoring Example ===")
    
    client = VulnerabilityDetectionClient()
    
    # Health check
    health = client.health_check()
    print(f"Health Status: {health}")
    
    # System status
    status = client.get_system_status()
    print(f"System Status: {json.dumps(status, indent=2)}")
    
    # Metrics
    metrics = client.get_metrics()
    print(f"Metrics: {json.dumps(metrics, indent=2)}")
    
    return {
        "health": health,
        "status": status,
        "metrics": metrics
    }

async def example_async_analysis():
    """Example: Asynchronous contract analysis"""
    print("\n=== Asynchronous Analysis Example ===")
    
    async with aiohttp.ClientSession() as session:
        # Multiple contracts to analyze concurrently
        contracts = [
            "contract Async1 { function test() public {} }",
            "contract Async2 { function test() public {} }",
            "contract Async3 { function test() public {} }"
        ]
        
        async def analyze_contract(session, contract_code, index):
            request_data = {
                "contract_code": contract_code,
                "model_type": "ensemble"
            }
            
            async with session.post(
                f"{API_BASE_URL}/api/v1/analyze",
                json=request_data
            ) as response:
                result = await response.json()
                print(f"Contract {index}: {result.get('success', False)}")
                return result
        
        # Run analyses concurrently
        tasks = [
            analyze_contract(session, contract, i) 
            for i, contract in enumerate(contracts)
        ]
        
        results = await asyncio.gather(*tasks)
        print(f"Async Results: {len(results)} analyses completed")
        
        return results

def example_error_handling():
    """Example: Error handling and edge cases"""
    print("\n=== Error Handling Example ===")
    
    client = VulnerabilityDetectionClient()
    
    # Test cases
    test_cases = [
        {
            "name": "Empty contract code",
            "contract_code": "",
            "expected_error": True
        },
        {
            "name": "Invalid contract code",
            "contract_code": "This is not Solidity code",
            "expected_error": True
        },
        {
            "name": "Invalid model type",
            "contract_code": "contract Test { function test() public {} }",
            "model_type": "invalid_model",
            "expected_error": True
        },
        {
            "name": "Valid contract",
            "contract_code": "contract Test { function test() public {} }",
            "expected_error": False
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        
        try:
            result = client.analyze_contract(
                contract_code=test_case["contract_code"],
                model_type=test_case.get("model_type", "ensemble")
            )
            
            if test_case["expected_error"]:
                print(f"Expected error, got: {result}")
            else:
                print(f"Success: {result.get('success', False)}")
        
        except Exception as e:
            print(f"Exception: {e}")

def main():
    """Main function to run all examples"""
    print("Smart Contract Vulnerability Detection API - Usage Examples")
    print("=" * 60)
    
    try:
        # Check API health first
        client = VulnerabilityDetectionClient()
        health = client.health_check()
        print(f"API Health: {health.get('status', 'unknown')}")
        
        if health.get('status') not in ['healthy', 'degraded']:
            print("API is not healthy. Please start the API server first.")
            return
        
        # Run examples
        example_safe_contract()
        example_vulnerable_contract()
        example_batch_analysis()
        example_model_comparison()
        example_system_monitoring()
        example_error_handling()
        
        # Run async example
        asyncio.run(example_async_analysis())
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure the API server is running on http://localhost:8000")

if __name__ == "__main__":
    main()
