"""
Regression Testing Framework for Smart Contract Vulnerability Detection

This module provides comprehensive regression testing against known vulnerable contracts
to validate the accuracy and consistency of AI audit results.
"""

import os
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import pandas as pd
from datetime import datetime

from benchmarking.static_analysis import (
    StaticAnalysisBenchmark, 
    VulnerabilityFinding, 
    create_vulnerability_finding_from_ai
)

logger = logging.getLogger(__name__)


@dataclass
class KnownVulnerability:
    """Represents a known vulnerability in a test contract."""
    contract_name: str
    vulnerability_type: str
    severity: str
    location: str
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    description: str = ""
    expected_confidence: float = 0.8  # Minimum expected confidence
    cve_reference: Optional[str] = None
    exploit_scenario: Optional[str] = None


@dataclass
class RegressionTestResult:
    """Results from a regression test."""
    contract_name: str
    test_timestamp: datetime
    ai_detected: bool
    ai_confidence: float
    ai_severity: str
    expected_vulnerability: KnownVulnerability
    false_positive: bool = False
    false_negative: bool = False
    severity_match: bool = False
    location_match: bool = False
    analysis_time: float = 0.0
    error_message: Optional[str] = None


class RegressionTestSuite:
    """Regression testing suite for vulnerability detection."""
    
    def __init__(self, test_contracts_dir: str = "tests/regression/contracts"):
        self.test_contracts_dir = Path(test_contracts_dir)
        self.benchmark = StaticAnalysisBenchmark()
        self.known_vulnerabilities: List[KnownVulnerability] = []
        self.test_results: List[RegressionTestResult] = []
        
        # Create test contracts directory if it doesn't exist
        self.test_contracts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load known vulnerabilities
        self._load_known_vulnerabilities()
    
    def _load_known_vulnerabilities(self):
        """Load known vulnerabilities from configuration."""
        vulnerabilities_config = [
            {
                "contract_name": "reentrancy_vulnerable.sol",
                "vulnerability_type": "reentrancy",
                "severity": "high",
                "location": "withdraw",
                "line_number": 25,
                "function_name": "withdraw",
                "description": "Reentrancy vulnerability in withdraw function",
                "expected_confidence": 0.9,
                "cve_reference": "CVE-2016-10709",
                "exploit_scenario": "Attacker can drain contract by calling withdraw recursively"
            },
            {
                "contract_name": "integer_overflow.sol",
                "vulnerability_type": "integer_overflow",
                "severity": "high",
                "location": "transfer",
                "line_number": 15,
                "function_name": "transfer",
                "description": "Integer overflow in transfer function",
                "expected_confidence": 0.85,
                "cve_reference": "CVE-2018-10299",
                "exploit_scenario": "Attacker can overflow balance to bypass checks"
            },
            {
                "contract_name": "unchecked_call.sol",
                "vulnerability_type": "unchecked_call",
                "severity": "medium",
                "location": "external_call",
                "line_number": 30,
                "function_name": "externalCall",
                "description": "Unchecked external call return value",
                "expected_confidence": 0.8,
                "cve_reference": None,
                "exploit_scenario": "Silent failure of external calls"
            },
            {
                "contract_name": "timestamp_dependency.sol",
                "vulnerability_type": "timestamp_dependency",
                "severity": "medium",
                "location": "randomize",
                "line_number": 20,
                "function_name": "randomize",
                "description": "Timestamp dependency for randomness",
                "expected_confidence": 0.75,
                "cve_reference": None,
                "exploit_scenario": "Miner can manipulate block timestamp"
            },
            {
                "contract_name": "tx_origin.sol",
                "vulnerability_type": "tx_origin",
                "severity": "medium",
                "location": "authorize",
                "line_number": 10,
                "function_name": "authorize",
                "description": "Use of tx.origin for authorization",
                "expected_confidence": 0.8,
                "cve_reference": None,
                "exploit_scenario": "Phishing attack through malicious contract"
            }
        ]
        
        for vuln_config in vulnerabilities_config:
            vulnerability = KnownVulnerability(**vuln_config)
            self.known_vulnerabilities.append(vulnerability)
    
    def create_test_contracts(self):
        """Create test contracts with known vulnerabilities."""
        contracts = {
            "reentrancy_vulnerable.sol": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract ReentrancyVulnerable {
    mapping(address => uint256) public balances;
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable: external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;  // State update after external call
    }
}
""",
            "integer_overflow.sol": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract IntegerOverflow {
    mapping(address => uint256) public balances;
    
    function transfer(address to, uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable: potential integer overflow
        balances[msg.sender] -= amount;
        balances[to] += amount;  // Could overflow if amount is very large
    }
}
""",
            "unchecked_call.sol": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract UncheckedCall {
    function externalCall(address target) public {
        // Vulnerable: unchecked external call
        target.call(abi.encodeWithSignature("execute()"));
        // No check of return value
    }
}
""",
            "timestamp_dependency.sol": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TimestampDependency {
    uint256 public randomSeed;
    
    function randomize() public {
        // Vulnerable: timestamp dependency
        randomSeed = uint256(keccak256(abi.encodePacked(block.timestamp, msg.sender)));
    }
}
""",
            "tx_origin.sol": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract TxOrigin {
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    function authorize() public {
        // Vulnerable: use of tx.origin
        require(tx.origin == owner, "Not authorized");
        // Function logic here
    }
}
"""
        }
        
        for filename, content in contracts.items():
            contract_path = self.test_contracts_dir / filename
            with open(contract_path, 'w') as f:
                f.write(content)
        
        logger.info(f"Created {len(contracts)} test contracts in {self.test_contracts_dir}")
    
    def run_regression_tests(self, ai_analyzer_func) -> List[RegressionTestResult]:
        """
        Run regression tests against known vulnerable contracts.
        
        Args:
            ai_analyzer_func: Function that takes contract path and returns AI analysis results
            
        Returns:
            List of regression test results
        """
        logger.info("Starting regression tests...")
        
        # Ensure test contracts exist
        if not any(self.test_contracts_dir.glob("*.sol")):
            self.create_test_contracts()
        
        results = []
        
        for vulnerability in self.known_vulnerabilities:
            contract_path = self.test_contracts_dir / vulnerability.contract_name
            
            if not contract_path.exists():
                logger.warning(f"Contract {contract_path} not found, skipping test")
                continue
            
            logger.info(f"Testing {vulnerability.contract_name} for {vulnerability.vulnerability_type}")
            
            try:
                # Run AI analysis
                start_time = time.time()
                ai_results = ai_analyzer_func(str(contract_path))
                analysis_time = time.time() - start_time
                
                # Convert AI results to VulnerabilityFinding objects
                ai_findings = []
                for result in ai_results:
                    if isinstance(result, dict):
                        ai_findings.append(create_vulnerability_finding_from_ai(result))
                    else:
                        ai_findings.append(result)
                
                # Check if vulnerability was detected
                detected = self._check_vulnerability_detected(ai_findings, vulnerability)
                
                # Create test result
                test_result = RegressionTestResult(
                    contract_name=vulnerability.contract_name,
                    test_timestamp=datetime.now(),
                    ai_detected=detected["detected"],
                    ai_confidence=detected["confidence"],
                    ai_severity=detected["severity"],
                    expected_vulnerability=vulnerability,
                    false_positive=detected["detected"] and not self._is_expected_vulnerability(ai_findings, vulnerability),
                    false_negative=not detected["detected"] and vulnerability.expected_confidence > 0.5,
                    severity_match=detected["severity"] == vulnerability.severity,
                    location_match=detected["location_match"],
                    analysis_time=analysis_time
                )
                
                results.append(test_result)
                logger.info(f"Test completed for {vulnerability.contract_name}: {'PASS' if not test_result.false_negative else 'FAIL'}")
                
            except Exception as e:
                logger.error(f"Error testing {vulnerability.contract_name}: {str(e)}")
                error_result = RegressionTestResult(
                    contract_name=vulnerability.contract_name,
                    test_timestamp=datetime.now(),
                    ai_detected=False,
                    ai_confidence=0.0,
                    ai_severity="unknown",
                    expected_vulnerability=vulnerability,
                    false_negative=True,
                    analysis_time=0.0,
                    error_message=str(e)
                )
                results.append(error_result)
        
        self.test_results = results
        return results
    
    def _check_vulnerability_detected(self, ai_findings: List[VulnerabilityFinding], 
                                    expected_vuln: KnownVulnerability) -> Dict[str, Any]:
        """Check if the expected vulnerability was detected by AI."""
        detected = {
            "detected": False,
            "confidence": 0.0,
            "severity": "unknown",
            "location_match": False
        }
        
        for finding in ai_findings:
            # Check if vulnerability type matches
            if finding.vulnerability_type.lower() == expected_vuln.vulnerability_type.lower():
                detected["detected"] = True
                detected["confidence"] = max(detected["confidence"], finding.confidence)
                detected["severity"] = finding.severity
                
                # Check location match
                if expected_vuln.function_name:
                    detected["location_match"] = (
                        finding.function_name == expected_vuln.function_name or
                        expected_vuln.function_name in finding.location
                    )
                else:
                    detected["location_match"] = expected_vuln.location in finding.location
        
        return detected
    
    def _is_expected_vulnerability(self, ai_findings: List[VulnerabilityFinding], 
                                 expected_vuln: KnownVulnerability) -> bool:
        """Check if AI findings contain the expected vulnerability."""
        for finding in ai_findings:
            if (finding.vulnerability_type.lower() == expected_vuln.vulnerability_type.lower() and
                finding.confidence >= expected_vuln.expected_confidence):
                return True
        return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive regression test report."""
        if not self.test_results:
            return {"error": "No test results available"}
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results if not result.false_negative)
        failed_tests = sum(1 for result in self.test_results if result.false_negative)
        false_positives = sum(1 for result in self.test_results if result.false_positive)
        
        # Calculate accuracy metrics
        accuracy = passed_tests / total_tests if total_tests > 0 else 0.0
        precision = passed_tests / (passed_tests + false_positives) if (passed_tests + false_positives) > 0 else 0.0
        
        # Calculate average analysis time
        avg_analysis_time = sum(result.analysis_time for result in self.test_results) / total_tests
        
        # Group results by vulnerability type
        vulnerability_stats = {}
        for result in self.test_results:
            vuln_type = result.expected_vulnerability.vulnerability_type
            if vuln_type not in vulnerability_stats:
                vulnerability_stats[vuln_type] = {
                    "total": 0,
                    "detected": 0,
                    "false_negatives": 0,
                    "avg_confidence": 0.0
                }
            
            vulnerability_stats[vuln_type]["total"] += 1
            if result.ai_detected:
                vulnerability_stats[vuln_type]["detected"] += 1
            if result.false_negative:
                vulnerability_stats[vuln_type]["false_negatives"] += 1
            
            vulnerability_stats[vuln_type]["avg_confidence"] += result.ai_confidence
        
        # Calculate averages
        for vuln_type in vulnerability_stats:
            stats = vulnerability_stats[vuln_type]
            stats["detection_rate"] = stats["detected"] / stats["total"] if stats["total"] > 0 else 0.0
            stats["avg_confidence"] = stats["avg_confidence"] / stats["total"] if stats["total"] > 0 else 0.0
        
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "false_positives": false_positives,
                "accuracy": accuracy,
                "precision": precision,
                "average_analysis_time": avg_analysis_time
            },
            "vulnerability_statistics": vulnerability_stats,
            "detailed_results": [asdict(result) for result in self.test_results],
            "test_timestamp": datetime.now().isoformat(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not self.test_results:
            return ["No test results available for recommendations"]
        
        # Check for high false negative rate
        false_negative_rate = sum(1 for r in self.test_results if r.false_negative) / len(self.test_results)
        if false_negative_rate > 0.3:
            recommendations.append("High false negative rate detected. Consider improving model sensitivity.")
        
        # Check for high false positive rate
        false_positive_rate = sum(1 for r in self.test_results if r.false_positive) / len(self.test_results)
        if false_positive_rate > 0.2:
            recommendations.append("High false positive rate detected. Consider improving model specificity.")
        
        # Check for severity mismatches
        severity_mismatches = sum(1 for r in self.test_results if not r.severity_match and r.ai_detected)
        if severity_mismatches > 0:
            recommendations.append("Severity classification needs improvement.")
        
        # Check analysis time
        avg_time = sum(r.analysis_time for r in self.test_results) / len(self.test_results)
        if avg_time > 10.0:
            recommendations.append("Analysis time is high. Consider optimizing model performance.")
        
        if not recommendations:
            recommendations.append("Test results look good! Model performance is within acceptable ranges.")
        
        return recommendations
    
    def save_results(self, output_path: str = "benchmarking/regression_results.json"):
        """Save test results to file."""
        report = self.generate_report()
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Convert all datetime objects
        report_str = json.dumps(report, default=convert_datetime, indent=2)
        
        with open(output_path, 'w') as f:
            f.write(report_str)
        
        logger.info(f"Regression test results saved to {output_path}")
    
    def export_to_csv(self, output_path: str = "benchmarking/regression_results.csv"):
        """Export test results to CSV format."""
        if not self.test_results:
            logger.warning("No test results to export")
            return
        
        # Convert results to DataFrame
        data = []
        for result in self.test_results:
            data.append({
                "contract_name": result.contract_name,
                "test_timestamp": result.test_timestamp.isoformat(),
                "expected_vulnerability_type": result.expected_vulnerability.vulnerability_type,
                "expected_severity": result.expected_vulnerability.severity,
                "expected_confidence": result.expected_vulnerability.expected_confidence,
                "ai_detected": result.ai_detected,
                "ai_confidence": result.ai_confidence,
                "ai_severity": result.ai_severity,
                "false_positive": result.false_positive,
                "false_negative": result.false_negative,
                "severity_match": result.severity_match,
                "location_match": result.location_match,
                "analysis_time": result.analysis_time,
                "error_message": result.error_message
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        logger.info(f"Regression test results exported to {output_path}")


def run_regression_tests(ai_analyzer_func, output_dir: str = "benchmarking") -> Dict[str, Any]:
    """
    Convenience function to run regression tests.
    
    Args:
        ai_analyzer_func: Function that takes contract path and returns AI analysis results
        output_dir: Directory to save results
        
    Returns:
        Test report dictionary
    """
    test_suite = RegressionTestSuite()
    results = test_suite.run_regression_tests(ai_analyzer_func)
    report = test_suite.generate_report()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    test_suite.save_results(f"{output_dir}/regression_results.json")
    test_suite.export_to_csv(f"{output_dir}/regression_results.csv")
    
    return report
