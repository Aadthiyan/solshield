"""
Performance Benchmarking Module

This module provides comprehensive performance benchmarking for the smart contract
vulnerability detection system, including accuracy validation and performance metrics.
"""

import os
import time
import json
import logging
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from memory_profiler import profile
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test."""
    test_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    throughput: Optional[float] = None
    latency: Optional[float] = None
    error_rate: Optional[float] = None
    timestamp: str = ""


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking tests."""
    test_duration: int = 300  # seconds
    concurrent_users: int = 10
    test_contracts: List[str] = None
    api_endpoint: str = "http://localhost:8000"
    model_paths: Dict[str, str] = None
    output_dir: str = "benchmarking/results"
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    enable_accuracy_validation: bool = True


class PerformanceMonitor:
    """Monitors system performance during benchmarking."""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = []
        self.start_time = None
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.monitoring = True
        self.metrics = []
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_used_gb = memory.used / (1024**3)
                
                # Get process metrics
                process = psutil.Process()
                process_memory = process.memory_info().rss / (1024**2)  # MB
                process_cpu = process.cpu_percent()
                
                metric = {
                    "timestamp": time.time() - self.start_time,
                    "system_cpu": cpu_percent,
                    "system_memory_percent": memory_percent,
                    "system_memory_gb": memory_used_gb,
                    "process_memory_mb": process_memory,
                    "process_cpu": process_cpu
                }
                
                self.metrics.append(metric)
                
            except Exception as e:
                logger.warning(f"Error in monitoring loop: {e}")
            
            time.sleep(1)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        if not self.metrics:
            return {}
        
        df = pd.DataFrame(self.metrics)
        
        return {
            "duration": df["timestamp"].max(),
            "avg_system_cpu": df["system_cpu"].mean(),
            "max_system_cpu": df["system_cpu"].max(),
            "avg_system_memory": df["system_memory_percent"].mean(),
            "max_system_memory": df["system_memory_percent"].max(),
            "avg_process_memory": df["process_memory_mb"].mean(),
            "max_process_memory": df["process_memory_mb"].max(),
            "avg_process_cpu": df["process_cpu"].mean(),
            "max_process_cpu": df["process_cpu"].max(),
            "total_samples": len(self.metrics)
        }


class AccuracyValidator:
    """Validates accuracy of vulnerability detection against known benchmarks."""
    
    def __init__(self, benchmark_dataset_path: str = "data/benchmarks/accuracy_benchmark.json"):
        self.benchmark_dataset_path = Path(benchmark_dataset_path)
        self.benchmark_data = self._load_benchmark_data()
    
    def _load_benchmark_data(self) -> Dict[str, Any]:
        """Load benchmark dataset for accuracy validation."""
        if not self.benchmark_dataset_path.exists():
            # Create sample benchmark data
            sample_data = {
                "contracts": [
                    {
                        "id": "vulnerable_1",
                        "code": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VulnerableContract {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable: external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
}
""",
                        "expected_vulnerabilities": [
                            {
                                "type": "reentrancy",
                                "severity": "high",
                                "location": "withdraw function",
                                "confidence": 0.9
                            }
                        ]
                    },
                    {
                        "id": "safe_1",
                        "code": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract SafeContract {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Safe: state update before external call
        balances[msg.sender] -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
""",
                        "expected_vulnerabilities": []
                    }
                ]
            }
            
            # Save sample data
            self.benchmark_dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.benchmark_dataset_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info(f"Created sample benchmark dataset at {self.benchmark_dataset_path}")
        
        with open(self.benchmark_dataset_path, 'r') as f:
            return json.load(f)
    
    def validate_accuracy(self, api_endpoint: str) -> Dict[str, Any]:
        """
        Validate accuracy against benchmark dataset.
        
        Args:
            api_endpoint: API endpoint for vulnerability analysis
            
        Returns:
            Accuracy validation results
        """
        logger.info("Starting accuracy validation...")
        
        results = {
            "total_contracts": len(self.benchmark_data["contracts"]),
            "correct_predictions": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "true_negatives": 0,
            "detailed_results": [],
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0
        }
        
        for contract_data in self.benchmark_data["contracts"]:
            try:
                # Analyze contract
                response = requests.post(
                    f"{api_endpoint}/analyze",
                    json={"contract_code": contract_data["code"]},
                    timeout=30
                )
                
                if response.status_code == 200:
                    analysis_result = response.json()
                    detected_vulnerabilities = analysis_result.get("vulnerabilities", [])
                    expected_vulnerabilities = contract_data["expected_vulnerabilities"]
                    
                    # Compare results
                    comparison = self._compare_vulnerabilities(
                        detected_vulnerabilities,
                        expected_vulnerabilities
                    )
                    
                    results["detailed_results"].append({
                        "contract_id": contract_data["id"],
                        "expected": expected_vulnerabilities,
                        "detected": detected_vulnerabilities,
                        "comparison": comparison
                    })
                    
                    # Update metrics
                    results["true_positives"] += comparison["true_positives"]
                    results["false_positives"] += comparison["false_positives"]
                    results["true_negatives"] += comparison["true_negatives"]
                    results["false_negatives"] += comparison["false_negatives"]
                    
                else:
                    logger.error(f"API request failed for contract {contract_data['id']}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error analyzing contract {contract_data['id']}: {e}")
        
        # Calculate final metrics
        total_predictions = results["true_positives"] + results["false_positives"] + results["true_negatives"] + results["false_negatives"]
        
        if total_predictions > 0:
            results["accuracy"] = (results["true_positives"] + results["true_negatives"]) / total_predictions
        
        if results["true_positives"] + results["false_positives"] > 0:
            results["precision"] = results["true_positives"] / (results["true_positives"] + results["false_positives"])
        
        if results["true_positives"] + results["false_negatives"] > 0:
            results["recall"] = results["true_positives"] / (results["true_positives"] + results["false_negatives"])
        
        if results["precision"] + results["recall"] > 0:
            results["f1_score"] = 2 * (results["precision"] * results["recall"]) / (results["precision"] + results["recall"])
        
        logger.info(f"Accuracy validation completed: {results['accuracy']:.3f}")
        return results
    
    def _compare_vulnerabilities(self, detected: List[Dict], expected: List[Dict]) -> Dict[str, int]:
        """Compare detected vulnerabilities with expected ones."""
        detected_types = {vuln.get("type", "").lower() for vuln in detected}
        expected_types = {vuln.get("type", "").lower() for vuln in expected}
        
        true_positives = len(detected_types.intersection(expected_types))
        false_positives = len(detected_types - expected_types)
        false_negatives = len(expected_types - detected_types)
        true_negatives = 1 if not detected_types and not expected_types else 0
        
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives
        }


class LoadTester:
    """Performs load testing on the vulnerability detection API."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.results = []
        self.monitor = PerformanceMonitor()
    
    def run_load_test(self) -> Dict[str, Any]:
        """Run comprehensive load test."""
        logger.info(f"Starting load test with {self.config.concurrent_users} concurrent users")
        
        # Start performance monitoring
        self.monitor.start_monitoring()
        
        start_time = time.time()
        
        try:
            # Run concurrent requests
            with ThreadPoolExecutor(max_workers=self.config.concurrent_users) as executor:
                futures = []
                
                for i in range(self.config.concurrent_users):
                    future = executor.submit(self._single_user_test, f"user_{i}")
                    futures.append(future)
                
                # Collect results
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        self.results.append(result)
                    except Exception as e:
                        logger.error(f"User test failed: {e}")
                        self.results.append({
                            "user_id": "unknown",
                            "success": False,
                            "error": str(e),
                            "response_time": 0
                        })
            
            total_time = time.time() - start_time
            
        finally:
            # Stop performance monitoring
            self.monitor.stop_monitoring()
        
        # Analyze results
        analysis = self._analyze_load_test_results(total_time)
        
        return analysis
    
    def _single_user_test(self, user_id: str) -> Dict[str, Any]:
        """Simulate a single user's behavior."""
        user_results = {
            "user_id": user_id,
            "requests": [],
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_response_time": 0,
            "avg_response_time": 0
        }
        
        end_time = time.time() + self.config.test_duration
        
        while time.time() < end_time:
            try:
                # Select random contract for testing
                test_contract = self._get_random_test_contract()
                
                # Make API request
                request_start = time.time()
                
                response = requests.post(
                    f"{self.config.api_endpoint}/analyze",
                    json={"contract_code": test_contract},
                    timeout=30
                )
                
                request_time = time.time() - request_start
                
                request_result = {
                    "timestamp": time.time(),
                    "response_time": request_time,
                    "status_code": response.status_code,
                    "success": response.status_code == 200
                }
                
                user_results["requests"].append(request_result)
                user_results["total_requests"] += 1
                user_results["total_response_time"] += request_time
                
                if request_result["success"]:
                    user_results["successful_requests"] += 1
                else:
                    user_results["failed_requests"] += 1
                
                # Add delay between requests
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Request failed for {user_id}: {e}")
                user_results["failed_requests"] += 1
        
        # Calculate averages
        if user_results["total_requests"] > 0:
            user_results["avg_response_time"] = user_results["total_response_time"] / user_results["total_requests"]
        
        return user_results
    
    def _get_random_test_contract(self) -> str:
        """Get a random test contract."""
        if self.config.test_contracts:
            import random
            return random.choice(self.config.test_contracts)
        
        # Default test contract
        return """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TestContract {
    uint256 public value;
    
    function setValue(uint256 _value) public {
        value = _value;
    }
    
    function getValue() public view returns (uint256) {
        return value;
    }
}
"""
    
    def _analyze_load_test_results(self, total_time: float) -> Dict[str, Any]:
        """Analyze load test results."""
        if not self.results:
            return {"error": "No test results available"}
        
        # Aggregate metrics
        total_requests = sum(r["total_requests"] for r in self.results)
        successful_requests = sum(r["successful_requests"] for r in self.results)
        failed_requests = sum(r["failed_requests"] for r in self.results)
        
        # Response time analysis
        all_response_times = []
        for result in self.results:
            all_response_times.extend([req["response_time"] for req in result["requests"]])
        
        # Performance monitoring summary
        perf_summary = self.monitor.get_summary()
        
        analysis = {
            "test_duration": total_time,
            "concurrent_users": self.config.concurrent_users,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "requests_per_second": total_requests / total_time if total_time > 0 else 0,
            "response_time_stats": {
                "min": min(all_response_times) if all_response_times else 0,
                "max": max(all_response_times) if all_response_times else 0,
                "avg": np.mean(all_response_times) if all_response_times else 0,
                "median": np.median(all_response_times) if all_response_times else 0,
                "p95": np.percentile(all_response_times, 95) if all_response_times else 0,
                "p99": np.percentile(all_response_times, 99) if all_response_times else 0
            },
            "performance_monitoring": perf_summary,
            "user_results": self.results
        }
        
        return analysis


class BenchmarkRunner:
    """Main benchmark runner that orchestrates all performance tests."""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.accuracy_validator = AccuracyValidator()
        self.load_tester = LoadTester(config)
        self.monitor = PerformanceMonitor()
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark."""
        logger.info("Starting comprehensive performance benchmark...")
        
        benchmark_results = {
            "benchmark_timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "tests": {},
            "summary": {},
            "success": True
        }
        
        try:
            # 1. Accuracy Validation
            if self.config.enable_accuracy_validation:
                logger.info("Running accuracy validation...")
                accuracy_results = self.accuracy_validator.validate_accuracy(self.config.api_endpoint)
                benchmark_results["tests"]["accuracy_validation"] = accuracy_results
            
            # 2. Load Testing
            logger.info("Running load testing...")
            load_test_results = self.load_tester.run_load_test()
            benchmark_results["tests"]["load_testing"] = load_test_results
            
            # 3. Memory Profiling
            if self.config.enable_memory_profiling:
                logger.info("Running memory profiling...")
                memory_results = self._run_memory_profiling()
                benchmark_results["tests"]["memory_profiling"] = memory_results
            
            # 4. CPU Profiling
            if self.config.enable_cpu_profiling:
                logger.info("Running CPU profiling...")
                cpu_results = self._run_cpu_profiling()
                benchmark_results["tests"]["cpu_profiling"] = cpu_results
            
            # Generate summary
            benchmark_results["summary"] = self._generate_summary(benchmark_results["tests"])
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            benchmark_results["success"] = False
            benchmark_results["error"] = str(e)
        
        # Save results
        self._save_results(benchmark_results)
        
        # Generate reports
        self._generate_reports(benchmark_results)
        
        logger.info("Comprehensive benchmark completed")
        return benchmark_results
    
    def _run_memory_profiling(self) -> Dict[str, Any]:
        """Run memory profiling tests."""
        # This would typically use memory_profiler or similar tools
        # For now, we'll simulate memory profiling
        
        memory_results = {
            "peak_memory_mb": 0,
            "average_memory_mb": 0,
            "memory_growth_rate": 0,
            "memory_leaks_detected": False,
            "profiling_details": {}
        }
        
        # Simulate memory profiling
        import psutil
        process = psutil.Process()
        
        memory_samples = []
        for _ in range(10):
            memory_samples.append(process.memory_info().rss / (1024**2))
            time.sleep(0.1)
        
        memory_results["peak_memory_mb"] = max(memory_samples)
        memory_results["average_memory_mb"] = np.mean(memory_samples)
        
        return memory_results
    
    def _run_cpu_profiling(self) -> Dict[str, Any]:
        """Run CPU profiling tests."""
        # This would typically use cProfile or similar tools
        # For now, we'll simulate CPU profiling
        
        cpu_results = {
            "cpu_usage_percent": 0,
            "cpu_intensive_functions": [],
            "profiling_details": {}
        }
        
        # Simulate CPU profiling
        cpu_samples = []
        for _ in range(10):
            cpu_samples.append(psutil.cpu_percent(interval=0.1))
        
        cpu_results["cpu_usage_percent"] = np.mean(cpu_samples)
        
        return cpu_results
    
    def _generate_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            "overall_success": True,
            "key_metrics": {},
            "recommendations": []
        }
        
        # Extract key metrics
        if "accuracy_validation" in test_results:
            acc = test_results["accuracy_validation"]
            summary["key_metrics"]["accuracy"] = acc.get("accuracy", 0)
            summary["key_metrics"]["precision"] = acc.get("precision", 0)
            summary["key_metrics"]["recall"] = acc.get("recall", 0)
            summary["key_metrics"]["f1_score"] = acc.get("f1_score", 0)
        
        if "load_testing" in test_results:
            load = test_results["load_testing"]
            summary["key_metrics"]["requests_per_second"] = load.get("requests_per_second", 0)
            summary["key_metrics"]["success_rate"] = load.get("success_rate", 0)
            summary["key_metrics"]["avg_response_time"] = load.get("response_time_stats", {}).get("avg", 0)
        
        # Generate recommendations
        if summary["key_metrics"].get("accuracy", 0) < 0.8:
            summary["recommendations"].append("Accuracy is below 80%. Consider retraining models.")
        
        if summary["key_metrics"].get("success_rate", 0) < 0.95:
            summary["recommendations"].append("Success rate is below 95%. Check API stability.")
        
        if summary["key_metrics"].get("avg_response_time", 0) > 5.0:
            summary["recommendations"].append("Average response time is high. Consider optimization.")
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save full results
        results_file = self.output_dir / f"benchmark_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def _generate_reports(self, results: Dict[str, Any]):
        """Generate visual reports."""
        try:
            self._generate_performance_charts(results)
            self._generate_summary_report(results)
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    def _generate_performance_charts(self, results: Dict[str, Any]):
        """Generate performance visualization charts."""
        if "load_testing" not in results["tests"]:
            return
        
        load_results = results["tests"]["load_testing"]
        
        # Response time distribution
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        response_times = []
        for user_result in load_results.get("user_results", []):
            response_times.extend([req["response_time"] for req in user_result["requests"]])
        
        if response_times:
            plt.hist(response_times, bins=50, alpha=0.7)
            plt.title("Response Time Distribution")
            plt.xlabel("Response Time (seconds)")
            plt.ylabel("Frequency")
        
        # Requests per second over time
        plt.subplot(2, 2, 2)
        # This would require more detailed timing data
        plt.title("Requests Per Second Over Time")
        plt.xlabel("Time")
        plt.ylabel("RPS")
        
        # Success rate by user
        plt.subplot(2, 2, 3)
        user_success_rates = []
        user_ids = []
        for user_result in load_results.get("user_results", []):
            if user_result["total_requests"] > 0:
                success_rate = user_result["successful_requests"] / user_result["total_requests"]
                user_success_rates.append(success_rate)
                user_ids.append(user_result["user_id"])
        
        if user_success_rates:
            plt.bar(user_ids, user_success_rates)
            plt.title("Success Rate by User")
            plt.xlabel("User ID")
            plt.ylabel("Success Rate")
            plt.xticks(rotation=45)
        
        # Performance metrics summary
        plt.subplot(2, 2, 4)
        metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
        values = []
        
        if "accuracy_validation" in results["tests"]:
            acc = results["tests"]["accuracy_validation"]
            values = [
                acc.get("accuracy", 0),
                acc.get("precision", 0),
                acc.get("recall", 0),
                acc.get("f1_score", 0)
            ]
        
        if values:
            plt.bar(metrics, values)
            plt.title("Accuracy Metrics")
            plt.ylabel("Score")
            plt.ylim(0, 1)
        
        plt.tight_layout()
        
        # Save chart
        chart_file = self.output_dir / f"performance_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Performance charts saved to {chart_file}")
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate text summary report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_summary_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Smart Contract Vulnerability Detection - Performance Benchmark Report\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Benchmark Date: {results['benchmark_timestamp']}\n")
            f.write(f"Overall Success: {results['success']}\n\n")
            
            # Key metrics
            if "summary" in results and "key_metrics" in results["summary"]:
                f.write("Key Metrics:\n")
                f.write("-" * 20 + "\n")
                for metric, value in results["summary"]["key_metrics"].items():
                    f.write(f"{metric}: {value:.3f}\n")
                f.write("\n")
            
            # Recommendations
            if "summary" in results and "recommendations" in results["summary"]:
                f.write("Recommendations:\n")
                f.write("-" * 20 + "\n")
                for i, rec in enumerate(results["summary"]["recommendations"], 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")
            
            # Test details
            f.write("Test Details:\n")
            f.write("-" * 20 + "\n")
            for test_name, test_results in results["tests"].items():
                f.write(f"\n{test_name.upper()}:\n")
                if isinstance(test_results, dict):
                    for key, value in test_results.items():
                        if not isinstance(value, (dict, list)):
                            f.write(f"  {key}: {value}\n")
        
        logger.info(f"Summary report saved to {report_file}")


def run_performance_benchmark(config_path: str = None) -> Dict[str, Any]:
    """
    Convenience function to run performance benchmark.
    
    Args:
        config_path: Path to benchmark configuration file
        
    Returns:
        Benchmark results
    """
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = BenchmarkConfig(**config_data)
    else:
        config = BenchmarkConfig()
    
    benchmark_runner = BenchmarkRunner(config)
    return benchmark_runner.run_comprehensive_benchmark()


if __name__ == "__main__":
    # Example usage
    config = BenchmarkConfig(
        test_duration=60,  # 1 minute test
        concurrent_users=5,
        api_endpoint="http://localhost:8000"
    )
    
    runner = BenchmarkRunner(config)
    results = runner.run_comprehensive_benchmark()
    
    print("Benchmark completed!")
    print(f"Success: {results['success']}")
    if "summary" in results:
        print(f"Key metrics: {results['summary']['key_metrics']}")
