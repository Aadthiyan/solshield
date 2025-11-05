#!/usr/bin/env python3
"""
Performance Testing Script

This module provides comprehensive performance testing for the smart contract
vulnerability detection API using various testing approaches.
"""

import asyncio
import aiohttp
import time
import statistics
import json
import random
from typing import List, Dict, Any, Tuple
import argparse
import sys
from pathlib import Path

# Add project directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

class PerformanceTester:
    """Performance testing class for the API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.results = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def test_single_request(self, contract_code: str, model_type: str = "ensemble") -> Dict[str, Any]:
        """Test a single request"""
        request_data = {
            "contract_code": contract_code,
            "model_type": model_type,
            "include_optimization_suggestions": True,
            "include_explanation": True
        }
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/v1/analyze",
                json=request_data
            ) as response:
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status == 200:
                    data = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "status_code": response.status,
                        "report_id": data.get("report_id"),
                        "processing_time": data.get("estimated_processing_time", 0)
                    }
                else:
                    return {
                        "success": False,
                        "response_time": response_time,
                        "status_code": response.status,
                        "error": await response.text()
                    }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    async def test_concurrent_requests(self, 
                                     contract_codes: List[str], 
                                     concurrency: int = 10) -> List[Dict[str, Any]]:
        """Test concurrent requests"""
        semaphore = asyncio.Semaphore(concurrency)
        
        async def limited_request(contract_code: str):
            async with semaphore:
                return await self.test_single_request(contract_code)
        
        tasks = [limited_request(code) for code in contract_codes]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    "success": False,
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def test_batch_requests(self, 
                                contract_codes: List[str], 
                                batch_size: int = 5) -> List[Dict[str, Any]]:
        """Test batch requests"""
        results = []
        
        # Split contracts into batches
        for i in range(0, len(contract_codes), batch_size):
            batch_contracts = contract_codes[i:i + batch_size]
            
            # Create batch request
            contracts_data = []
            for contract_code in batch_contracts:
                contracts_data.append({
                    "contract_code": contract_code,
                    "model_type": "ensemble"
                })
            
            request_data = {
                "contracts": contracts_data,
                "batch_id": f"batch_{int(time.time())}_{i}"
            }
            
            start_time = time.time()
            
            try:
                async with self.session.post(
                    f"{self.base_url}/api/v1/analyze/batch",
                    json=request_data
                ) as response:
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    if response.status == 200:
                        data = await response.json()
                        results.append({
                            "success": True,
                            "response_time": response_time,
                            "status_code": response.status,
                            "batch_id": data.get("batch_id"),
                            "contract_count": len(batch_contracts)
                        })
                    else:
                        results.append({
                            "success": False,
                            "response_time": response_time,
                            "status_code": response.status,
                            "error": await response.text()
                        })
            except Exception as e:
                end_time = time.time()
                results.append({
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                })
        
        return results
    
    async def test_health_endpoints(self) -> Dict[str, Any]:
        """Test health and status endpoints"""
        endpoints = [
            "/health",
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/metrics"
        ]
        
        results = {}
        
        for endpoint in endpoints:
            start_time = time.time()
            
            try:
                async with self.session.get(f"{self.base_url}{endpoint}") as response:
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    results[endpoint] = {
                        "success": response.status == 200,
                        "response_time": response_time,
                        "status_code": response.status
                    }
            except Exception as e:
                end_time = time.time()
                results[endpoint] = {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e)
                }
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze test results"""
        if not results:
            return {"error": "No results to analyze"}
        
        # Filter successful results
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", False)]
        
        if not successful_results:
            return {
                "total_requests": len(results),
                "successful_requests": 0,
                "failed_requests": len(failed_results),
                "success_rate": 0.0,
                "error": "No successful requests"
            }
        
        # Calculate statistics
        response_times = [r["response_time"] for r in successful_results]
        
        analysis = {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(results),
            "response_time_stats": {
                "min": min(response_times),
                "max": max(response_times),
                "mean": statistics.mean(response_times),
                "median": statistics.median(response_times),
                "std": statistics.stdev(response_times) if len(response_times) > 1 else 0
            },
            "throughput": len(successful_results) / sum(response_times) if response_times else 0
        }
        
        # Add error analysis if there are failures
        if failed_results:
            error_types = {}
            for result in failed_results:
                error = result.get("error", "Unknown error")
                error_types[error] = error_types.get(error, 0) + 1
            
            analysis["error_analysis"] = error_types
        
        return analysis

class LoadTestRunner:
    """Load test runner with different scenarios"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_contracts = self._load_test_contracts()
    
    def _load_test_contracts(self) -> List[str]:
        """Load test contracts"""
        return [
            # Simple contract
            """
            contract SimpleContract {
                function test() public {
                    require(msg.sender != address(0));
                }
            }
            """,
            
            # Medium complexity contract
            """
            contract MediumContract {
                uint256 public balance;
                address public owner;
                
                modifier onlyOwner() {
                    require(msg.sender == owner, "Not the owner");
                    _;
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
            """,
            
            # Complex contract
            """
            contract ComplexContract {
                struct User {
                    uint256 balance;
                    bool isActive;
                    uint256 lastActivity;
                }
                
                mapping(address => User) public users;
                address public owner;
                
                modifier onlyOwner() {
                    require(msg.sender == owner, "Not the owner");
                    _;
                }
                
                function createUser() public {
                    users[msg.sender] = User({
                        balance: 0,
                        isActive: true,
                        lastActivity: block.timestamp
                    });
                }
                
                function deposit() public payable {
                    require(users[msg.sender].isActive, "User not active");
                    users[msg.sender].balance += msg.value;
                    users[msg.sender].lastActivity = block.timestamp;
                }
                
                function withdraw(uint256 amount) public {
                    require(users[msg.sender].isActive, "User not active");
                    require(users[msg.sender].balance >= amount, "Insufficient balance");
                    users[msg.sender].balance -= amount;
                    users[msg.sender].lastActivity = block.timestamp;
                    payable(msg.sender).transfer(amount);
                }
            }
            """,
            
            # Vulnerable contract
            """
            contract VulnerableContract {
                mapping(address => uint256) public balances;
                
                function withdraw() public {
                    uint256 amount = balances[msg.sender];
                    require(amount > 0, "No balance");
                    balances[msg.sender] = 0;
                    msg.sender.transfer(amount);
                }
                
                function deposit() public payable {
                    balances[msg.sender] += msg.value;
                }
            }
            """
        ]
    
    async def run_concurrent_test(self, 
                                num_requests: int = 100, 
                                concurrency: int = 10) -> Dict[str, Any]:
        """Run concurrent request test"""
        print(f"Running concurrent test: {num_requests} requests, {concurrency} concurrent")
        
        # Generate test contracts
        test_contracts = [random.choice(self.test_contracts) for _ in range(num_requests)]
        
        async with PerformanceTester(self.base_url) as tester:
            results = await tester.test_concurrent_requests(test_contracts, concurrency)
            analysis = tester.analyze_results(results)
        
        return analysis
    
    async def run_batch_test(self, 
                           num_batches: int = 20, 
                           batch_size: int = 5) -> Dict[str, Any]:
        """Run batch request test"""
        print(f"Running batch test: {num_batches} batches, {batch_size} contracts per batch")
        
        # Generate test contracts
        test_contracts = [random.choice(self.test_contracts) for _ in range(num_batches * batch_size)]
        
        async with PerformanceTester(self.base_url) as tester:
            results = await tester.test_batch_requests(test_contracts, batch_size)
            analysis = tester.analyze_results(results)
        
        return analysis
    
    async def run_health_test(self) -> Dict[str, Any]:
        """Run health endpoint test"""
        print("Running health endpoint test")
        
        async with PerformanceTester(self.base_url) as tester:
            results = await tester.test_health_endpoints()
        
        return results
    
    async def run_stress_test(self, 
                             duration_seconds: int = 60, 
                             requests_per_second: int = 10) -> Dict[str, Any]:
        """Run stress test"""
        print(f"Running stress test: {duration_seconds}s, {requests_per_second} req/s")
        
        start_time = time.time()
        results = []
        
        async with PerformanceTester(self.base_url) as tester:
            while time.time() - start_time < duration_seconds:
                # Create batch of requests
                batch_contracts = [random.choice(self.test_contracts) for _ in range(requests_per_second)]
                batch_results = await tester.test_concurrent_requests(batch_contracts, requests_per_second)
                results.extend(batch_results)
                
                # Wait for next second
                await asyncio.sleep(1)
        
        # Analyze results
        async with PerformanceTester(self.base_url) as tester:
            analysis = tester.analyze_results(results)
        
        return analysis

async def main():
    """Main function for running performance tests"""
    parser = argparse.ArgumentParser(description="Performance testing for smart contract vulnerability detection API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--test", choices=["concurrent", "batch", "health", "stress", "all"], 
                       default="all", help="Test type to run")
    parser.add_argument("--requests", type=int, default=100, help="Number of requests for concurrent test")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--duration", type=int, default=60, help="Duration for stress test (seconds)")
    parser.add_argument("--output", help="Output file for results")
    
    args = parser.parse_args()
    
    runner = LoadTestRunner(args.url)
    results = {}
    
    try:
        if args.test in ["concurrent", "all"]:
            print("\n=== Concurrent Request Test ===")
            concurrent_results = await runner.run_concurrent_test(args.requests, args.concurrency)
            results["concurrent"] = concurrent_results
            print(f"Results: {json.dumps(concurrent_results, indent=2)}")
        
        if args.test in ["batch", "all"]:
            print("\n=== Batch Request Test ===")
            batch_results = await runner.run_batch_test(20, 5)
            results["batch"] = batch_results
            print(f"Results: {json.dumps(batch_results, indent=2)}")
        
        if args.test in ["health", "all"]:
            print("\n=== Health Endpoint Test ===")
            health_results = await runner.run_health_test()
            results["health"] = health_results
            print(f"Results: {json.dumps(health_results, indent=2)}")
        
        if args.test in ["stress", "all"]:
            print("\n=== Stress Test ===")
            stress_results = await runner.run_stress_test(args.duration, 10)
            results["stress"] = stress_results
            print(f"Results: {json.dumps(stress_results, indent=2)}")
        
        # Save results to file
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
        
        # Print summary
        print("\n=== Test Summary ===")
        for test_name, test_results in results.items():
            if isinstance(test_results, dict) and "success_rate" in test_results:
                print(f"{test_name}: {test_results['success_rate']:.2%} success rate")
            elif isinstance(test_results, dict):
                print(f"{test_name}: {len(test_results)} endpoints tested")
    
    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
