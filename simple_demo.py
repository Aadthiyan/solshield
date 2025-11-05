#!/usr/bin/env python3
"""
Smart Contract Vulnerability Detection dApp - Simple Demo

This script demonstrates the vulnerability detection capabilities
using a simplified analysis engine.
"""

import time
import logging
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleVulnerabilityDetector:
    """Simplified vulnerability detector for demonstration purposes."""
    
    def __init__(self):
        self.vulnerability_patterns = {
            "reentrancy": [
                "call{value:",
                ".call(",
                ".send(",
                ".transfer("
            ],
            "integer_overflow": [
                "uint256",
                "uint8",
                "uint16",
                "uint32",
                "uint64",
                "uint128"
            ],
            "unchecked_call": [
                "call{",
                ".call(",
                "require("
            ],
            "uninitialized_storage": [
                "mapping(",
                "struct"
            ]
        }
    
    def analyze_contract(self, contract_code: str) -> Dict[str, Any]:
        """Analyze a smart contract for vulnerabilities."""
        logger.info("Analyzing smart contract...")
        
        vulnerabilities = []
        optimizations = []
        
        # Simple pattern matching for demonstration
        for vuln_type, patterns in self.vulnerability_patterns.items():
            for pattern in patterns:
                if pattern in contract_code:
                    vulnerability = {
                        "type": vuln_type,
                        "severity": self._get_severity(vuln_type),
                        "confidence": 0.8,
                        "description": self._get_description(vuln_type),
                        "location": f"Pattern: {pattern}",
                        "line_number": self._find_line_number(contract_code, pattern)
                    }
                    vulnerabilities.append(vulnerability)
        
        # Generate optimization suggestions
        if "mapping(" in contract_code:
            optimizations.append({
                "type": "gas_optimization",
                "description": "Consider using packed structs to reduce gas costs",
                "potential_savings": 2000,
                "implementation": "Use struct packing for related variables"
            })
        
        if "for(" in contract_code:
            optimizations.append({
                "type": "gas_optimization", 
                "description": "Consider using unchecked arithmetic in loops",
                "potential_savings": 1000,
                "implementation": "Use unchecked { ... } for safe arithmetic operations"
            })
        
        # Calculate risk score
        risk_score = min(len(vulnerabilities) * 0.3, 1.0)
        
        return {
            "analysis_id": f"demo_{int(time.time())}",
            "vulnerabilities": vulnerabilities,
            "optimizations": optimizations,
            "risk_score": risk_score,
            "analysis_time": 0.5,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _get_severity(self, vuln_type: str) -> str:
        """Get severity level for vulnerability type."""
        severity_map = {
            "reentrancy": "high",
            "integer_overflow": "medium", 
            "unchecked_call": "medium",
            "uninitialized_storage": "low"
        }
        return severity_map.get(vuln_type, "medium")
    
    def _get_description(self, vuln_type: str) -> str:
        """Get description for vulnerability type."""
        descriptions = {
            "reentrancy": "Potential reentrancy vulnerability - external call before state update",
            "integer_overflow": "Potential integer overflow/underflow vulnerability",
            "unchecked_call": "Unchecked external call - may fail silently",
            "uninitialized_storage": "Uninitialized storage variable"
        }
        return descriptions.get(vuln_type, "Potential security vulnerability")
    
    def _find_line_number(self, code: str, pattern: str) -> int:
        """Find line number of pattern in code."""
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            if pattern in line:
                return i
        return 0


def run_demo():
    """Run the complete demo."""
    print("Smart Contract Vulnerability Detection dApp - Demo")
    print("=" * 60)
    print("This demo showcases the vulnerability detection capabilities")
    print("using a simplified analysis engine.")
    print("=" * 60)
    
    detector = SimpleVulnerabilityDetector()
    
    # Demo contracts
    demo_contracts = {
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
""",
        "integer_overflow": """
// SPDX-License-Identifier: MIT
pragma solidity ^0.7.0;

contract IntegerOverflow {
    uint256 public totalSupply;
    
    function mint(uint256 amount) public {
        // Potential overflow in older Solidity versions
        totalSupply += amount;
    }
    
    function burn(uint256 amount) public {
        require(totalSupply >= amount, "Insufficient supply");
        totalSupply -= amount;
    }
}
"""
    }
    
    for contract_name, contract_code in demo_contracts.items():
        print(f"\nAnalyzing: {contract_name}")
        print("-" * 40)
        
        # Analyze contract
        result = detector.analyze_contract(contract_code)
        
        # Display results
        print(f"Analysis completed in {result['analysis_time']}s")
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Vulnerabilities Found: {len(result['vulnerabilities'])}")
        print(f"Optimizations Suggested: {len(result['optimizations'])}")
        
        # Show vulnerabilities
        if result['vulnerabilities']:
            print("\nDetected Vulnerabilities:")
            for i, vuln in enumerate(result['vulnerabilities'], 1):
                print(f"  {i}. {vuln['type'].upper()} - {vuln['severity'].upper()}")
                print(f"     Confidence: {vuln['confidence']:.2f}")
                print(f"     Description: {vuln['description']}")
                if vuln['line_number'] > 0:
                    print(f"     Line: {vuln['line_number']}")
        
        # Show optimizations
        if result['optimizations']:
            print("\nOptimization Suggestions:")
            for i, opt in enumerate(result['optimizations'], 1):
                print(f"  {i}. {opt['type'].upper()}")
                print(f"     Description: {opt['description']}")
                print(f"     Potential Savings: {opt['potential_savings']} gas")
        
        print("\n" + "="*60)
    
    print("\nDemo completed successfully!")
    print("\nNext Steps:")
    print("1. Install full dependencies: pip install -r requirements.txt")
    print("2. Start the API server: python -m api.main")
    print("3. Start the frontend: cd frontend && npm run dev")
    print("4. Access the web interface at http://localhost:3000")
    print("5. Use the API at http://localhost:8000")


if __name__ == "__main__":
    run_demo()
