#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection dApp - Demo Script

This script demonstrates the key features of the smart contract vulnerability
detection system, including AI analysis, static analysis comparison, and
performance benchmarking.
"""

import os
import sys
import time
import json
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from benchmarking.static_analysis import StaticAnalysisTools
from benchmarking.benchmark_report import BenchmarkReportGenerator
from benchmarking.performance_benchmark import BenchmarkRunner, BenchmarkConfig
from deployment.qie_deployment import QIEDeploymentManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VulnerabilityDetectionDemo:
    """Main demo class for the vulnerability detection system."""
    
    def __init__(self, api_endpoint: str = "http://localhost:8000"):
        self.api_endpoint = api_endpoint
        self.demo_contracts = self._load_demo_contracts()
        self.results = []
    
    def _load_demo_contracts(self) -> Dict[str, str]:
        """Load demo contracts for testing."""
        return {
            "reentrancy_vulnerable": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReentrancyVulnerable {
    mapping(address => uint256) public balances;
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable: external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
""",
            "integer_overflow": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.7.0;

contract IntegerOverflow {
    uint256 public totalSupply;
    
    function mint(uint256 amount) public {
        // Vulnerable: potential overflow in older Solidity versions
        totalSupply += amount;
    }
    
    function burn(uint256 amount) public {
        require(totalSupply >= amount, "Insufficient supply");
        totalSupply -= amount;
    }
}
""",
            "unchecked_call": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UncheckedCall {
    function transfer(address to, uint256 amount) public {
        // Vulnerable: unchecked external call
        to.call{value: amount}("");
    }
    
    function safeTransfer(address to, uint256 amount) public {
        // Safe: checked external call
        (bool success, ) = to.call{value: amount}("");
        require(success, "Transfer failed");
    }
}
""",
            "safe_contract": """
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
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
"""
        }
    
    def check_api_availability(self) -> bool:
        """Check if the API is available."""
        try:
            response = requests.get(f"{self.api_endpoint}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API not available: {e}")
            return False
    
    def demo_ai_analysis(self) -> Dict[str, Any]:
        """Demonstrate AI-powered vulnerability analysis."""
        print("\n" + "="*60)
        print("🤖 AI-POWERED VULNERABILITY ANALYSIS DEMO")
        print("="*60)
        
        results = {}
        
        for contract_name, contract_code in self.demo_contracts.items():
            print(f"\n📋 Analyzing: {contract_name}")
            print("-" * 40)
            
            try:
                # Make API request
                start_time = time.time()
                response = requests.post(
                    f"{self.api_endpoint}/analyze",
                    json={"contract_code": contract_code},
                    timeout=30
                )
                analysis_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Display results
                    vulnerabilities = result.get("vulnerabilities", [])
                    optimizations = result.get("optimizations", [])
                    risk_score = result.get("risk_score", 0)
                    
                    print(f"✅ Analysis completed in {analysis_time:.2f}s")
                    print(f"🎯 Risk Score: {risk_score:.2f}")
                    print(f"🚨 Vulnerabilities Found: {len(vulnerabilities)}")
                    print(f"⚡ Optimizations Suggested: {len(optimizations)}")
                    
                    # Show vulnerabilities
                    if vulnerabilities:
                        print("\n🔍 Detected Vulnerabilities:")
                        for i, vuln in enumerate(vulnerabilities, 1):
                            print(f"  {i}. {vuln.get('type', 'Unknown')} - {vuln.get('severity', 'Unknown')}")
                            print(f"     Confidence: {vuln.get('confidence', 0):.2f}")
                            print(f"     Location: {vuln.get('location', 'Unknown')}")
                    
                    # Show optimizations
                    if optimizations:
                        print("\n💡 Optimization Suggestions:")
                        for i, opt in enumerate(optimizations, 1):
                            print(f"  {i}. {opt.get('type', 'Unknown')}")
                            print(f"     Potential Savings: {opt.get('potential_savings', 0)} gas")
                    
                    results[contract_name] = {
                        "success": True,
                        "analysis_time": analysis_time,
                        "vulnerabilities": vulnerabilities,
                        "optimizations": optimizations,
                        "risk_score": risk_score
                    }
                    
                else:
                    print(f"❌ Analysis failed: HTTP {response.status_code}")
                    results[contract_name] = {
                        "success": False,
                        "error": f"HTTP {response.status_code}"
                    }
                    
            except Exception as e:
                print(f"❌ Analysis error: {e}")
                results[contract_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def demo_static_analysis(self) -> Dict[str, Any]:
        """Demonstrate static analysis tools comparison."""
        print("\n" + "="*60)
        print("🔧 STATIC ANALYSIS TOOLS COMPARISON DEMO")
        print("="*60)
        
        static_tools = StaticAnalysisTools()
        results = {}
        
        for contract_name, contract_code in self.demo_contracts.items():
            print(f"\n📋 Analyzing: {contract_name}")
            print("-" * 40)
            
            try:
                # Run Slither
                print("🔍 Running Slither...")
                slither_result = static_tools.run_slither(contract_code)
                print(f"   Vulnerabilities: {len(slither_result.get('vulnerabilities', []))}")
                print(f"   Execution Time: {slither_result.get('execution_time', 0):.2f}s")
                
                # Run Mythril
                print("🔍 Running Mythril...")
                mythril_result = static_tools.run_mythril(contract_code)
                print(f"   Vulnerabilities: {len(mythril_result.get('vulnerabilities', []))}")
                print(f"   Execution Time: {mythril_result.get('execution_time', 0):.2f}s")
                
                results[contract_name] = {
                    "slither": slither_result,
                    "mythril": mythril_result
                }
                
            except Exception as e:
                print(f"❌ Static analysis error: {e}")
                results[contract_name] = {
                    "error": str(e)
                }
        
        return results
    
    def demo_benchmark_comparison(self) -> Dict[str, Any]:
        """Demonstrate benchmark comparison between AI and static analysis."""
        print("\n" + "="*60)
        print("📊 BENCHMARK COMPARISON DEMO")
        print("="*60)
        
        try:
            # Generate benchmark report
            generator = BenchmarkReportGenerator()
            results = generator.run_benchmark(self.api_endpoint)
            
            print(f"✅ Benchmark completed for {len(results)} contracts")
            
            # Generate and save report
            report = generator.generate_report(results)
            
            print(f"📈 Report generated with summary:")
            summary = report["summary"]
            print(f"   Best Tool: {summary['best_overall_tool']}")
            print(f"   Average Accuracy: {summary['average_accuracy']:.3f}")
            
            # Show tool performance
            print("\n🏆 Tool Performance:")
            for tool_name, metrics in summary["tool_performance"].items():
                print(f"   {tool_name.upper()}:")
                print(f"     Accuracy: {metrics['accuracy']:.3f}")
                print(f"     Precision: {metrics['precision']:.3f}")
                print(f"     Recall: {metrics['recall']:.3f}")
                print(f"     F1-Score: {metrics['f1_score']:.3f}")
            
            # Show recommendations
            if report["recommendations"]:
                print("\n💡 Recommendations:")
                for i, rec in enumerate(report["recommendations"], 1):
                    print(f"   {i}. {rec}")
            
            return report
            
        except Exception as e:
            print(f"❌ Benchmark error: {e}")
            return {"error": str(e)}
    
    def demo_performance_benchmark(self) -> Dict[str, Any]:
        """Demonstrate performance benchmarking."""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE BENCHMARK DEMO")
        print("="*60)
        
        try:
            # Configure benchmark
            config = BenchmarkConfig(
                test_duration=30,  # 30 seconds for demo
                concurrent_users=3,
                api_endpoint=self.api_endpoint,
                enable_accuracy_validation=True,
                enable_memory_profiling=True,
                enable_cpu_profiling=True
            )
            
            # Run benchmark
            runner = BenchmarkRunner(config)
            results = runner.run_comprehensive_benchmark()
            
            if results["success"]:
                print("✅ Performance benchmark completed")
                
                # Show key metrics
                if "summary" in results and "key_metrics" in results["summary"]:
                    metrics = results["summary"]["key_metrics"]
                    print(f"\n📊 Key Metrics:")
                    print(f"   Accuracy: {metrics.get('accuracy', 0):.3f}")
                    print(f"   Requests/Second: {metrics.get('requests_per_second', 0):.1f}")
                    print(f"   Success Rate: {metrics.get('success_rate', 0):.3f}")
                    print(f"   Avg Response Time: {metrics.get('avg_response_time', 0):.3f}s")
                
                # Show recommendations
                if "summary" in results and "recommendations" in results["summary"]:
                    recommendations = results["summary"]["recommendations"]
                    if recommendations:
                        print("\n💡 Performance Recommendations:")
                        for i, rec in enumerate(recommendations, 1):
                            print(f"   {i}. {rec}")
            else:
                print(f"❌ Benchmark failed: {results.get('error', 'Unknown error')}")
            
            return results
            
        except Exception as e:
            print(f"❌ Performance benchmark error: {e}")
            return {"error": str(e)}
    
    def demo_qie_deployment(self) -> Dict[str, Any]:
        """Demonstrate QIE testnet deployment."""
        print("\n" + "="*60)
        print("🌐 QIE TESTNET DEPLOYMENT DEMO")
        print("="*60)
        
        try:
            # Check if QIE private key is available
            qie_private_key = os.getenv("QIE_PRIVATE_KEY")
            if not qie_private_key:
                print("⚠️  QIE_PRIVATE_KEY environment variable not set")
                print("   Skipping QIE deployment demo")
                return {"skipped": True, "reason": "No private key"}
            
            # Initialize deployment manager
            deployment_manager = QIEDeploymentManager()
            
            # Check deployment status
            print("🔍 Checking deployment status...")
            status = deployment_manager.get_deployment_status()
            
            if status["contract_deployed"]:
                print(f"✅ Contract already deployed at: {status['contract_address']}")
                
                # Test deployment
                print("🧪 Testing deployed contract...")
                test_results = deployment_manager.test_deployment()
                
                if test_results.get("overall_success"):
                    print("✅ Contract tests passed")
                else:
                    print(f"❌ Contract tests failed: {test_results.get('error', 'Unknown error')}")
                
                return {
                    "already_deployed": True,
                    "contract_address": status["contract_address"],
                    "test_results": test_results
                }
            else:
                print("🚀 Deploying contract to QIE testnet...")
                deployment_results = deployment_manager.deploy_dapp()
                
                if deployment_results["success"]:
                    contract_address = deployment_results["contracts"]["vulnerability_auditor"]["address"]
                    print(f"✅ Contract deployed at: {contract_address}")
                    
                    # Test deployment
                    print("🧪 Testing deployed contract...")
                    test_results = deployment_manager.test_deployment()
                    
                    return {
                        "deployed": True,
                        "contract_address": contract_address,
                        "deployment_results": deployment_results,
                        "test_results": test_results
                    }
                else:
                    print(f"❌ Deployment failed: {deployment_results.get('errors', ['Unknown error'])}")
                    return {
                        "deployment_failed": True,
                        "errors": deployment_results.get("errors", [])
                    }
            
        except Exception as e:
            print(f"❌ QIE deployment error: {e}")
            return {"error": str(e)}
    
    def run_full_demo(self) -> Dict[str, Any]:
        """Run the complete demo showcasing all features."""
        print("🚀 SMART CONTRACT VULNERABILITY DETECTION DAPP DEMO")
        print("=" * 60)
        print("This demo showcases the key features of our AI-powered")
        print("smart contract vulnerability detection system.")
        print("=" * 60)
        
        demo_results = {
            "demo_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_available": False,
            "ai_analysis": {},
            "static_analysis": {},
            "benchmark_comparison": {},
            "performance_benchmark": {},
            "qie_deployment": {}
        }
        
        # Check API availability
        print("\n🔍 Checking API availability...")
        if self.check_api_availability():
            print("✅ API is available and ready")
            demo_results["api_available"] = True
            
            # Run AI analysis demo
            demo_results["ai_analysis"] = self.demo_ai_analysis()
            
            # Run static analysis demo
            demo_results["static_analysis"] = self.demo_static_analysis()
            
            # Run benchmark comparison
            demo_results["benchmark_comparison"] = self.demo_benchmark_comparison()
            
            # Run performance benchmark
            demo_results["performance_benchmark"] = self.demo_performance_benchmark()
            
        else:
            print("❌ API is not available")
            print("   Please start the backend API first:")
            print("   python -m api.main")
        
        # Run QIE deployment demo (independent of API)
        demo_results["qie_deployment"] = self.demo_qie_deployment()
        
        # Save demo results
        self._save_demo_results(demo_results)
        
        # Print summary
        self._print_demo_summary(demo_results)
        
        return demo_results
    
    def _save_demo_results(self, results: Dict[str, Any]):
        """Save demo results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = Path("demo_results") / f"demo_results_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Demo results saved to: {results_file}")
    
    def _print_demo_summary(self, results: Dict[str, Any]):
        """Print demo summary."""
        print("\n" + "="*60)
        print("📋 DEMO SUMMARY")
        print("="*60)
        
        print(f"🕒 Demo Time: {results['demo_timestamp']}")
        print(f"🌐 API Available: {'✅' if results['api_available'] else '❌'}")
        
        if results["api_available"]:
            # AI Analysis summary
            ai_results = results["ai_analysis"]
            successful_analyses = sum(1 for r in ai_results.values() if r.get("success", False))
            print(f"🤖 AI Analyses: {successful_analyses}/{len(ai_results)} successful")
            
            # Static Analysis summary
            static_results = results["static_analysis"]
            print(f"🔧 Static Analyses: {len(static_results)} contracts analyzed")
            
            # Benchmark summary
            benchmark_results = results["benchmark_comparison"]
            if "summary" in benchmark_results:
                summary = benchmark_results["summary"]
                print(f"📊 Best Tool: {summary.get('best_overall_tool', 'Unknown')}")
                print(f"📈 Average Accuracy: {summary.get('average_accuracy', 0):.3f}")
            
            # Performance summary
            perf_results = results["performance_benchmark"]
            if perf_results.get("success"):
                print("⚡ Performance Benchmark: ✅ Completed")
            else:
                print("⚡ Performance Benchmark: ❌ Failed")
        
        # QIE Deployment summary
        qie_results = results["qie_deployment"]
        if qie_results.get("deployed"):
            print(f"🌐 QIE Deployment: ✅ Contract deployed")
        elif qie_results.get("already_deployed"):
            print(f"🌐 QIE Deployment: ✅ Already deployed")
        elif qie_results.get("skipped"):
            print(f"🌐 QIE Deployment: ⚠️ Skipped ({qie_results.get('reason', 'Unknown')})")
        else:
            print(f"🌐 QIE Deployment: ❌ Failed")
        
        print("\n🎉 Demo completed! Check the generated reports for detailed results.")


def main():
    """Main function to run the demo."""
    parser = argparse.ArgumentParser(description="Smart Contract Vulnerability Detection Demo")
    parser.add_argument(
        "--api-endpoint",
        default="http://localhost:8000",
        help="API endpoint URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--demo-type",
        choices=["full", "ai", "static", "benchmark", "performance", "qie"],
        default="full",
        help="Type of demo to run (default: full)"
    )
    
    args = parser.parse_args()
    
    # Initialize demo
    demo = VulnerabilityDetectionDemo(api_endpoint=args.api_endpoint)
    
    # Run selected demo
    if args.demo_type == "full":
        demo.run_full_demo()
    elif args.demo_type == "ai":
        demo.demo_ai_analysis()
    elif args.demo_type == "static":
        demo.demo_static_analysis()
    elif args.demo_type == "benchmark":
        demo.demo_benchmark_comparison()
    elif args.demo_type == "performance":
        demo.demo_performance_benchmark()
    elif args.demo_type == "qie":
        demo.demo_qie_deployment()


if __name__ == "__main__":
    main()
