#!/usr/bin/env python3
"""
Innovative Labeled Data Creation Using Proxy Signals

This module implements advanced data labeling techniques that use proxy signals
to identify safe code patterns and enhance vulnerability detection beyond
explicit labels. It includes:

1. Security Best Practice Detection
2. Proxy Signal Generation
3. Soft Label Creation
4. Data Augmentation with Proxy Labels
"""

import ast
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import json

@dataclass
class ProxySignal:
    """Represents a proxy signal for data labeling"""
    signal_type: str
    confidence: float
    source_pattern: str
    context: Dict[str, Any]
    line_number: int
    importance_weight: float = 1.0

@dataclass
class SecurityPattern:
    """Represents a security best practice pattern"""
    pattern_name: str
    regex_pattern: str
    safety_score: float
    description: str
    vulnerability_types: List[str]

class SecurityBestPracticeDetector:
    """Detects security best practices in smart contract code"""
    
    def __init__(self):
        self.security_patterns = self._initialize_security_patterns()
        self.safety_indicators = self._initialize_safety_indicators()
        self.vulnerability_indicators = self._initialize_vulnerability_indicators()
    
    def _initialize_security_patterns(self) -> List[SecurityPattern]:
        """Initialize security best practice patterns"""
        patterns = [
            SecurityPattern(
                pattern_name="checks_effects_interactions",
                regex_pattern=r"require\s*\([^)]+\)\s*;\s*[^;]*;\s*[^;]*\.call\s*\(",
                safety_score=0.9,
                description="Checks-Effects-Interactions pattern",
                vulnerability_types=["reentrancy"]
            ),
            SecurityPattern(
                pattern_name="access_control_modifier",
                regex_pattern=r"modifier\s+\w+\s*\([^)]*\)\s*\{\s*require\s*\(",
                safety_score=0.8,
                description="Access control modifier",
                vulnerability_types=["access_control"]
            ),
            SecurityPattern(
                pattern_name="safe_math_operations",
                regex_pattern=r"SafeMath\s*\.\s*(add|sub|mul|div)\s*\(",
                safety_score=0.85,
                description="SafeMath operations",
                vulnerability_types=["integer_overflow"]
            ),
            SecurityPattern(
                pattern_name="event_emission",
                regex_pattern=r"emit\s+\w+\s*\(",
                safety_score=0.7,
                description="Event emission for transparency",
                vulnerability_types=["front_running"]
            ),
            SecurityPattern(
                pattern_name="time_lock_mechanism",
                regex_pattern=r"require\s*\(\s*block\.timestamp\s*>\s*\w+\s*\+\s*\d+\s*\)",
                safety_score=0.8,
                description="Time lock mechanism",
                vulnerability_types=["timestamp_dependence"]
            ),
            SecurityPattern(
                pattern_name="multi_sig_validation",
                regex_pattern=r"require\s*\(\s*isValidSignature\s*\(",
                safety_score=0.9,
                description="Multi-signature validation",
                vulnerability_types=["access_control"]
            ),
            SecurityPattern(
                pattern_name="circuit_breaker",
                regex_pattern=r"require\s*\(\s*!paused\s*\)",
                safety_score=0.8,
                description="Circuit breaker pattern",
                vulnerability_types=["dos"]
            ),
            SecurityPattern(
                pattern_name="withdrawal_pattern",
                regex_pattern=r"mapping\s*\(\s*address\s*=>\s*uint256\s*\)\s*private\s+\w+;",
                safety_score=0.75,
                description="Secure withdrawal pattern",
                vulnerability_types=["reentrancy"]
            )
        ]
        return patterns
    
    def _initialize_safety_indicators(self) -> Dict[str, float]:
        """Initialize safety indicators with confidence scores"""
        return {
            "has_require_statements": 0.6,
            "has_assert_statements": 0.5,
            "has_events": 0.4,
            "has_modifiers": 0.7,
            "has_constructor": 0.3,
            "has_fallback_function": 0.2,
            "uses_safemath": 0.8,
            "has_pausable": 0.7,
            "has_ownable": 0.6,
            "has_reentrancy_guard": 0.9
        }
    
    def _initialize_vulnerability_indicators(self) -> Dict[str, float]:
        """Initialize vulnerability indicators with risk scores"""
        return {
            "has_external_calls": -0.8,
            "has_loops": -0.3,
            "has_recursive_calls": -0.9,
            "has_assembly": -0.6,
            "has_delegatecall": -0.9,
            "has_selfdestruct": -0.7,
            "has_timestamp_dependence": -0.5,
            "has_block_dependence": -0.4,
            "has_unchecked_arithmetic": -0.7,
            "has_dynamic_arrays": -0.2
        }
    
    def detect_proxy_signals(self, contract_code: str) -> List[ProxySignal]:
        """Detect proxy signals in contract code"""
        signals = []
        lines = contract_code.split('\n')
        
        # Detect security patterns
        for i, line in enumerate(lines):
            for pattern in self.security_patterns:
                if re.search(pattern.regex_pattern, line, re.IGNORECASE):
                    signal = ProxySignal(
                        signal_type="security_pattern",
                        confidence=pattern.safety_score,
                        source_pattern=pattern.pattern_name,
                        context={
                            "description": pattern.description,
                            "vulnerability_types": pattern.vulnerability_types
                        },
                        line_number=i + 1,
                        importance_weight=pattern.safety_score
                    )
                    signals.append(signal)
        
        # Detect safety indicators
        safety_score = self._calculate_safety_score(contract_code)
        if safety_score > 0.5:
            signal = ProxySignal(
                signal_type="safety_indicator",
                confidence=safety_score,
                source_pattern="overall_safety",
                context={"safety_score": safety_score},
                line_number=0,
                importance_weight=safety_score
            )
            signals.append(signal)
        
        # Detect vulnerability indicators
        vulnerability_score = self._calculate_vulnerability_score(contract_code)
        if vulnerability_score < -0.3:
            signal = ProxySignal(
                signal_type="vulnerability_indicator",
                confidence=abs(vulnerability_score),
                source_pattern="overall_vulnerability",
                context={"vulnerability_score": vulnerability_score},
                line_number=0,
                importance_weight=abs(vulnerability_score)
            )
            signals.append(signal)
        
        return signals
    
    def _calculate_safety_score(self, contract_code: str) -> float:
        """Calculate overall safety score for the contract"""
        score = 0.0
        total_indicators = 0
        
        for indicator, weight in self.safety_indicators.items():
            if self._has_indicator(contract_code, indicator):
                score += weight
            total_indicators += 1
        
        return score / total_indicators if total_indicators > 0 else 0.0
    
    def _calculate_vulnerability_score(self, contract_code: str) -> float:
        """Calculate overall vulnerability score for the contract"""
        score = 0.0
        total_indicators = 0
        
        for indicator, weight in self.vulnerability_indicators.items():
            if self._has_indicator(contract_code, indicator):
                score += weight
            total_indicators += 1
        
        return score / total_indicators if total_indicators > 0 else 0.0
    
    def _has_indicator(self, code: str, indicator: str) -> bool:
        """Check if code has a specific indicator"""
        patterns = {
            "has_require_statements": r"require\s*\(",
            "has_assert_statements": r"assert\s*\(",
            "has_events": r"event\s+\w+",
            "has_modifiers": r"modifier\s+\w+",
            "has_constructor": r"constructor\s*\(",
            "has_fallback_function": r"fallback\s*\(",
            "uses_safemath": r"SafeMath",
            "has_pausable": r"Pausable",
            "has_ownable": r"Ownable",
            "has_reentrancy_guard": r"ReentrancyGuard",
            "has_external_calls": r"\.call\s*\(",
            "has_loops": r"(for|while)\s*\(",
            "has_recursive_calls": r"this\.\w+\s*\(",
            "has_assembly": r"assembly\s*\{",
            "has_delegatecall": r"\.delegatecall\s*\(",
            "has_selfdestruct": r"selfdestruct\s*\(",
            "has_timestamp_dependence": r"block\.timestamp",
            "has_block_dependence": r"block\.(number|hash)",
            "has_unchecked_arithmetic": r"unchecked\s*\{",
            "has_dynamic_arrays": r"\[\]\s+\w+"
        }
        
        if indicator in patterns:
            return bool(re.search(patterns[indicator], code, re.IGNORECASE))
        
        return False

class ProxyLabelGenerator:
    """Generates proxy labels using multiple signal sources"""
    
    def __init__(self):
        self.detector = SecurityBestPracticeDetector()
        self.signal_weights = {
            "security_pattern": 0.4,
            "safety_indicator": 0.3,
            "vulnerability_indicator": 0.3
        }
    
    def generate_proxy_labels(self, contract_code: str, explicit_label: int) -> Dict[str, Any]:
        """Generate proxy labels for a contract"""
        # Detect proxy signals
        signals = self.detector.detect_proxy_signals(contract_code)
        
        # Calculate proxy label scores
        proxy_scores = self._calculate_proxy_scores(signals)
        
        # Generate soft labels
        soft_labels = self._generate_soft_labels(proxy_scores, explicit_label)
        
        # Generate augmented labels
        augmented_labels = self._generate_augmented_labels(contract_code, signals)
        
        return {
            "explicit_label": explicit_label,
            "proxy_signals": signals,
            "proxy_scores": proxy_scores,
            "soft_labels": soft_labels,
            "augmented_labels": augmented_labels,
            "confidence": self._calculate_confidence(signals)
        }
    
    def _calculate_proxy_scores(self, signals: List[ProxySignal]) -> Dict[str, float]:
        """Calculate proxy scores from signals"""
        scores = {
            "safety_score": 0.0,
            "vulnerability_score": 0.0,
            "complexity_score": 0.0,
            "security_score": 0.0
        }
        
        for signal in signals:
            if signal.signal_type == "security_pattern":
                scores["security_score"] += signal.confidence * signal.importance_weight
            elif signal.signal_type == "safety_indicator":
                scores["safety_score"] += signal.confidence * signal.importance_weight
            elif signal.signal_type == "vulnerability_indicator":
                scores["vulnerability_score"] += signal.confidence * signal.importance_weight
        
        # Normalize scores
        for key in scores:
            scores[key] = min(max(scores[key], 0.0), 1.0)
        
        return scores
    
    def _generate_soft_labels(self, proxy_scores: Dict[str, float], explicit_label: int) -> Dict[str, float]:
        """Generate soft labels combining explicit and proxy information"""
        # Base soft label from explicit label
        soft_labels = {
            "vulnerable": float(explicit_label),
            "safe": 1.0 - float(explicit_label)
        }
        
        # Adjust based on proxy scores
        if proxy_scores["security_score"] > 0.7:
            soft_labels["safe"] = min(soft_labels["safe"] + 0.2, 1.0)
            soft_labels["vulnerable"] = max(soft_labels["vulnerable"] - 0.2, 0.0)
        
        if proxy_scores["vulnerability_score"] > 0.5:
            soft_labels["vulnerable"] = min(soft_labels["vulnerable"] + 0.3, 1.0)
            soft_labels["safe"] = max(soft_labels["safe"] - 0.3, 0.0)
        
        # Add complexity-based adjustments
        if proxy_scores["complexity_score"] > 0.8:
            soft_labels["complex"] = 0.8
        else:
            soft_labels["complex"] = proxy_scores["complexity_score"]
        
        return soft_labels
    
    def _generate_augmented_labels(self, contract_code: str, signals: List[ProxySignal]) -> Dict[str, Any]:
        """Generate augmented labels for data augmentation"""
        augmented_labels = {
            "vulnerability_types": [],
            "severity_levels": [],
            "risk_factors": [],
            "security_patterns": []
        }
        
        # Extract vulnerability types from signals
        for signal in signals:
            if signal.signal_type == "security_pattern":
                augmented_labels["security_patterns"].append(signal.source_pattern)
                if "context" in signal.context and "vulnerability_types" in signal.context:
                    augmented_labels["vulnerability_types"].extend(signal.context["vulnerability_types"])
        
        # Determine severity levels
        vulnerability_count = len(augmented_labels["vulnerability_types"])
        if vulnerability_count == 0:
            augmented_labels["severity_levels"] = ["low"]
        elif vulnerability_count <= 2:
            augmented_labels["severity_levels"] = ["medium"]
        else:
            augmented_labels["severity_levels"] = ["high", "critical"]
        
        # Extract risk factors
        risk_factors = []
        if "has_external_calls" in contract_code.lower():
            risk_factors.append("external_calls")
        if "has_loops" in contract_code.lower():
            risk_factors.append("loops")
        if "has_assembly" in contract_code.lower():
            risk_factors.append("assembly")
        
        augmented_labels["risk_factors"] = risk_factors
        
        return augmented_labels
    
    def _calculate_confidence(self, signals: List[ProxySignal]) -> float:
        """Calculate confidence in proxy labels"""
        if not signals:
            return 0.0
        
        total_confidence = sum(signal.confidence * signal.importance_weight for signal in signals)
        total_weight = sum(signal.importance_weight for signal in signals)
        
        return total_confidence / total_weight if total_weight > 0 else 0.0

class DataAugmentationWithProxyLabels:
    """Augments data using proxy labels for better generalization"""
    
    def __init__(self):
        self.proxy_generator = ProxyLabelGenerator()
        self.augmentation_strategies = [
            "code_obfuscation",
            "variable_renaming",
            "comment_removal",
            "whitespace_normalization",
            "dead_code_insertion"
        ]
    
    def augment_dataset(self, contracts: List[Dict[str, Any]], 
                       augmentation_factor: int = 3) -> List[Dict[str, Any]]:
        """Augment dataset using proxy labels"""
        augmented_contracts = []
        
        for contract in contracts:
            # Original contract
            augmented_contracts.append(contract)
            
            # Generate proxy labels
            proxy_labels = self.proxy_generator.generate_proxy_labels(
                contract["code"], contract["label"]
            )
            
            # Create augmented versions
            for i in range(augmentation_factor):
                augmented_contract = self._create_augmented_contract(
                    contract, proxy_labels, strategy=self.augmentation_strategies[i % len(self.augmentation_strategies)]
                )
                augmented_contracts.append(augmented_contract)
        
        return augmented_contracts
    
    def _create_augmented_contract(self, original_contract: Dict[str, Any], 
                                 proxy_labels: Dict[str, Any], 
                                 strategy: str) -> Dict[str, Any]:
        """Create an augmented version of a contract"""
        augmented_contract = original_contract.copy()
        
        if strategy == "code_obfuscation":
            augmented_contract["code"] = self._obfuscate_code(original_contract["code"])
        elif strategy == "variable_renaming":
            augmented_contract["code"] = self._rename_variables(original_contract["code"])
        elif strategy == "comment_removal":
            augmented_contract["code"] = self._remove_comments(original_contract["code"])
        elif strategy == "whitespace_normalization":
            augmented_contract["code"] = self._normalize_whitespace(original_contract["code"])
        elif strategy == "dead_code_insertion":
            augmented_contract["code"] = self._insert_dead_code(original_contract["code"])
        
        # Add proxy label information
        augmented_contract["proxy_labels"] = proxy_labels
        augmented_contract["augmentation_strategy"] = strategy
        
        return augmented_contract
    
    def _obfuscate_code(self, code: str) -> str:
        """Obfuscate code while preserving functionality"""
        # Simple obfuscation: replace variable names with random names
        import random
        import string
        
        # Find variable names and replace them
        variable_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = set(re.findall(variable_pattern, code))
        
        obfuscated_code = code
        for var in variables:
            if len(var) > 2 and not var in ['function', 'contract', 'pragma', 'solidity']:
                new_name = ''.join(random.choices(string.ascii_lowercase, k=len(var)))
                obfuscated_code = re.sub(r'\b' + var + r'\b', new_name, obfuscated_code)
        
        return obfuscated_code
    
    def _rename_variables(self, code: str) -> str:
        """Rename variables to make code look different"""
        # Similar to obfuscation but more systematic
        return self._obfuscate_code(code)
    
    def _remove_comments(self, code: str) -> str:
        """Remove comments from code"""
        # Remove single-line comments
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in code"""
        # Replace multiple spaces with single space
        code = re.sub(r' +', ' ', code)
        # Remove trailing whitespace
        code = re.sub(r' +$', '', code, flags=re.MULTILINE)
        return code
    
    def _insert_dead_code(self, code: str) -> str:
        """Insert dead code that doesn't affect functionality"""
        dead_code = """
        // Dead code for augmentation
        function unusedFunction() private pure returns (uint256) {
            uint256 unused = 0;
            return unused;
        }
        """
        
        # Insert dead code before the last closing brace
        if '}' in code:
            last_brace = code.rfind('}')
            code = code[:last_brace] + dead_code + '\n' + code[last_brace:]
        
        return code

# Example usage and testing
if __name__ == "__main__":
    # Test proxy labeling system
    sample_contract = """
    pragma solidity ^0.8.0;
    
    contract SecureContract {
        mapping(address => uint256) private balances;
        
        modifier onlyOwner() {
            require(msg.sender == owner, "Not owner");
            _;
        }
        
        function deposit() public payable {
            require(msg.value > 0, "Invalid amount");
            balances[msg.sender] += msg.value;
            emit Deposit(msg.sender, msg.value);
        }
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            require(amount > 0, "Invalid amount");
            
            balances[msg.sender] -= amount;
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            
            emit Withdrawal(msg.sender, amount);
        }
    }
    """
    
    # Test proxy label generation
    generator = ProxyLabelGenerator()
    proxy_labels = generator.generate_proxy_labels(sample_contract, label=0)
    
    print("Proxy Labels Generated:")
    print(f"Explicit Label: {proxy_labels['explicit_label']}")
    print(f"Proxy Scores: {proxy_labels['proxy_scores']}")
    print(f"Soft Labels: {proxy_labels['soft_labels']}")
    print(f"Confidence: {proxy_labels['confidence']}")
    
    # Test data augmentation
    contracts = [{"code": sample_contract, "label": 0}]
    augmenter = DataAugmentationWithProxyLabels()
    augmented_contracts = augmenter.augment_dataset(contracts, augmentation_factor=2)
    
    print(f"\nOriginal contracts: {len(contracts)}")
    print(f"Augmented contracts: {len(augmented_contracts)}")
    print(f"Augmentation strategies used: {[c['augmentation_strategy'] for c in augmented_contracts[1:]]}")
