#!/usr/bin/env python3
"""
Inference Engine for Smart Contract Vulnerability Detection

This module provides the core inference engine that loads models and
performs vulnerability detection on smart contracts.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import hashlib
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

# Add project directories to path
sys.path.append(str(Path(__file__).parent.parent.parent / "models"))
sys.path.append(str(Path(__file__).parent.parent.parent / "evaluation"))

from models.codebert_model import CodeBERTTrainer
from models.gnn_model import GNNTrainer
from api.models.schemas import (
    VulnerabilityReport, VulnerabilityDetail, OptimizationSuggestion,
    ModelPrediction, VulnerabilityType, VulnerabilitySeverity, ModelType
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages loading and caching of ML models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = {}
        self.model_versions = {}
        self.load_times = {}
        self.last_used = {}
        self.performance_metrics = {}
        self._lock = threading.Lock()
    
    def load_model(self, model_type: ModelType, model_path: Optional[str] = None) -> bool:
        """Load a specific model"""
        try:
            with self._lock:
                if model_type in self.models:
                    logger.info(f"Model {model_type} already loaded")
                    return True
                
                start_time = time.time()
                
                if model_type == ModelType.CODEBERT:
                    trainer = CodeBERTTrainer()
                    if model_path:
                        trainer.load_model(model_path)
                    else:
                        # Load from default path
                        default_path = self.model_dir / "codebert_output"
                        if default_path.exists():
                            trainer.load_model(str(default_path))
                        else:
                            logger.warning(f"CodeBERT model not found at {default_path}")
                            return False
                    
                    self.models[model_type] = trainer
                
                elif model_type == ModelType.GNN:
                    trainer = GNNTrainer(input_dim=22)
                    if model_path:
                        trainer.load_model(model_path)
                    else:
                        # Load from default path
                        default_path = self.model_dir / "gnn_output"
                        if default_path.exists():
                            trainer.load_model(str(default_path))
                        else:
                            logger.warning(f"GNN model not found at {default_path}")
                            return False
                    
                    self.models[model_type] = trainer
                
                else:
                    logger.error(f"Unsupported model type: {model_type}")
                    return False
                
                load_time = time.time() - start_time
                self.load_times[model_type] = load_time
                self.last_used[model_type] = time.time()
                
                # Load model version info
                self._load_model_version(model_type, model_path)
                
                logger.info(f"Model {model_type} loaded successfully in {load_time:.2f}s")
                return True
                
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {e}")
            return False
    
    def _load_model_version(self, model_type: ModelType, model_path: Optional[str]):
        """Load model version information"""
        try:
            if model_path:
                config_path = Path(model_path) / "config.json"
            else:
                config_path = self.model_dir / f"{model_type.value}_output" / "config.json"
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                self.model_versions[model_type] = config.get('version', 'unknown')
            else:
                self.model_versions[model_type] = 'unknown'
        except Exception as e:
            logger.warning(f"Could not load version info for {model_type}: {e}")
            self.model_versions[model_type] = 'unknown'
    
    def get_model(self, model_type: ModelType):
        """Get a loaded model"""
        return self.models.get(model_type)
    
    def is_model_loaded(self, model_type: ModelType) -> bool:
        """Check if model is loaded"""
        return model_type in self.models
    
    def unload_model(self, model_type: ModelType):
        """Unload a model to free memory"""
        if model_type in self.models:
            del self.models[model_type]
            if model_type in self.load_times:
                del self.load_times[model_type]
            if model_type in self.last_used:
                del self.last_used[model_type]
            logger.info(f"Model {model_type} unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {}
        for model_type in self.models:
            info[model_type.value] = {
                'is_loaded': True,
                'load_time': self.load_times.get(model_type, 0),
                'last_used': self.last_used.get(model_type, 0),
                'version': self.model_versions.get(model_type, 'unknown')
            }
        return info

class VulnerabilityAnalyzer:
    """Analyzes smart contracts for vulnerabilities"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.optimization_patterns = self._load_optimization_patterns()
    
    def _load_vulnerability_patterns(self) -> Dict[str, Dict]:
        """Load vulnerability detection patterns"""
        return {
            'reentrancy': {
                'patterns': ['.call{value:', '.send(', '.transfer('],
                'context': 'external_call',
                'severity': VulnerabilitySeverity.HIGH
            },
            'integer_overflow': {
                'patterns': ['+', '-', '*', '/'],
                'context': 'arithmetic',
                'severity': VulnerabilitySeverity.HIGH
            },
            'access_control': {
                'patterns': ['selfdestruct(', 'suicide('],
                'context': 'destructive',
                'severity': VulnerabilitySeverity.HIGH
            },
            'unchecked_external_calls': {
                'patterns': ['.call(', '.send(', '.transfer('],
                'context': 'external_call',
                'severity': VulnerabilitySeverity.MEDIUM
            },
            'timestamp_dependence': {
                'patterns': ['block.timestamp', 'now'],
                'context': 'timestamp',
                'severity': VulnerabilitySeverity.MEDIUM
            },
            'tx_origin': {
                'patterns': ['tx.origin'],
                'context': 'authorization',
                'severity': VulnerabilitySeverity.MEDIUM
            }
        }
    
    def _load_optimization_patterns(self) -> Dict[str, Dict]:
        """Load gas optimization patterns"""
        return {
            'storage_optimization': {
                'patterns': ['uint256', 'mapping'],
                'suggestions': 'Consider using smaller data types and packed structs'
            },
            'loop_optimization': {
                'patterns': ['for (', 'while ('],
                'suggestions': 'Consider loop unrolling and batch operations'
            },
            'function_optimization': {
                'patterns': ['function'],
                'suggestions': 'Use view/pure functions when possible'
            }
        }
    
    async def analyze_contract(self, 
                             contract_code: str, 
                             model_type: ModelType = ModelType.ENSEMBLE) -> VulnerabilityReport:
        """Analyze a smart contract for vulnerabilities"""
        start_time = time.time()
        
        # Generate contract hash
        contract_hash = hashlib.sha256(contract_code.encode()).hexdigest()
        
        # Initialize report
        report = VulnerabilityReport(
            contract_hash=contract_hash,
            is_vulnerable=False,
            overall_confidence=0.0,
            risk_score=0.0,
            processing_time=0.0
        )
        
        try:
            # Run model predictions
            model_predictions = await self._run_model_predictions(contract_code, model_type)
            report.model_predictions = model_predictions
            
            # Analyze vulnerabilities
            vulnerabilities = await self._analyze_vulnerabilities(contract_code, model_predictions)
            report.vulnerabilities = vulnerabilities
            
            # Generate optimization suggestions
            if len(contract_code) > 100:  # Only for substantial contracts
                optimization_suggestions = await self._generate_optimization_suggestions(contract_code)
                report.optimization_suggestions = optimization_suggestions
            
            # Calculate overall assessment
            report.is_vulnerable = len(vulnerabilities) > 0
            report.overall_confidence = max([pred.confidence for pred in model_predictions], default=0.0)
            report.risk_score = self._calculate_risk_score(vulnerabilities)
            
            # Update model usage
            for pred in model_predictions:
                if pred.model_type in self.model_manager.models:
                    self.model_manager.last_used[pred.model_type] = time.time()
            
        except Exception as e:
            logger.error(f"Error analyzing contract: {e}")
            raise
        
        finally:
            report.processing_time = time.time() - start_time
        
        return report
    
    async def _run_model_predictions(self, 
                                   contract_code: str, 
                                   model_type: ModelType) -> List[ModelPrediction]:
        """Run model predictions on contract code"""
        predictions = []
        
        if model_type == ModelType.ENSEMBLE:
            # Run both models
            tasks = []
            if self.model_manager.is_model_loaded(ModelType.CODEBERT):
                tasks.append(self._predict_codebert(contract_code))
            if self.model_manager.is_model_loaded(ModelType.GNN):
                tasks.append(self._predict_gnn(contract_code))
            
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in results:
                    if isinstance(result, ModelPrediction):
                        predictions.append(result)
            else:
                # Fallback to rule-based analysis
                predictions.append(self._rule_based_prediction(contract_code))
        
        elif model_type == ModelType.CODEBERT:
            if self.model_manager.is_model_loaded(ModelType.CODEBERT):
                predictions.append(await self._predict_codebert(contract_code))
            else:
                predictions.append(self._rule_based_prediction(contract_code))
        
        elif model_type == ModelType.GNN:
            if self.model_manager.is_model_loaded(ModelType.GNN):
                predictions.append(await self._predict_gnn(contract_code))
            else:
                predictions.append(self._rule_based_prediction(contract_code))
        
        return predictions
    
    async def _predict_codebert(self, contract_code: str) -> ModelPrediction:
        """Run CodeBERT prediction"""
        start_time = time.time()
        
        try:
            trainer = self.model_manager.get_model(ModelType.CODEBERT)
            if not trainer:
                raise ValueError("CodeBERT model not loaded")
            
            # Make prediction
            predicted_classes, predicted_probs = trainer.predict([contract_code])
            
            is_vulnerable = predicted_classes[0] == 1
            confidence = float(predicted_probs[0][1]) if is_vulnerable else float(predicted_probs[0][0])
            
            # Determine vulnerability types based on confidence and patterns
            vulnerability_types = self._determine_vulnerability_types(contract_code, confidence)
            
            processing_time = time.time() - start_time
            
            return ModelPrediction(
                model_type=ModelType.CODEBERT,
                is_vulnerable=is_vulnerable,
                confidence=confidence,
                vulnerability_types=vulnerability_types,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"CodeBERT prediction failed: {e}")
            return self._rule_based_prediction(contract_code)
    
    async def _predict_gnn(self, contract_code: str) -> ModelPrediction:
        """Run GNN prediction"""
        start_time = time.time()
        
        try:
            trainer = self.model_manager.get_model(ModelType.GNN)
            if not trainer:
                raise ValueError("GNN model not loaded")
            
            # Make prediction
            predicted_classes, predicted_probs = trainer.predict([contract_code])
            
            is_vulnerable = predicted_classes[0] == 1
            confidence = float(predicted_probs[0][1]) if is_vulnerable else float(predicted_probs[0][0])
            
            # Determine vulnerability types
            vulnerability_types = self._determine_vulnerability_types(contract_code, confidence)
            
            processing_time = time.time() - start_time
            
            return ModelPrediction(
                model_type=ModelType.GNN,
                is_vulnerable=is_vulnerable,
                confidence=confidence,
                vulnerability_types=vulnerability_types,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"GNN prediction failed: {e}")
            return self._rule_based_prediction(contract_code)
    
    def _rule_based_prediction(self, contract_code: str) -> ModelPrediction:
        """Fallback rule-based prediction"""
        start_time = time.time()
        
        # Simple rule-based analysis
        vulnerability_indicators = [
            '.call{value:', '.send(', '.transfer(',
            'selfdestruct(', 'suicide(',
            'tx.origin', 'block.timestamp', 'now'
        ]
        
        vulnerability_count = sum(1 for pattern in vulnerability_indicators 
                                if pattern in contract_code)
        
        is_vulnerable = vulnerability_count > 0
        confidence = min(vulnerability_count * 0.2, 0.8) if is_vulnerable else 0.1
        
        vulnerability_types = self._determine_vulnerability_types(contract_code, confidence)
        
        processing_time = time.time() - start_time
        
        return ModelPrediction(
            model_type=ModelType.CODEBERT,  # Default to CodeBERT for rule-based
            is_vulnerable=is_vulnerable,
            confidence=confidence,
            vulnerability_types=vulnerability_types,
            processing_time=processing_time
        )
    
    def _determine_vulnerability_types(self, 
                                     contract_code: str, 
                                     confidence: float) -> List[VulnerabilityType]:
        """Determine vulnerability types based on code patterns"""
        vulnerability_types = []
        
        for vuln_type, pattern_info in self.vulnerability_patterns.items():
            for pattern in pattern_info['patterns']:
                if pattern in contract_code:
                    try:
                        vuln_enum = VulnerabilityType(vuln_type)
                        if vuln_enum not in vulnerability_types:
                            vulnerability_types.append(vuln_enum)
                    except ValueError:
                        continue
        
        if not vulnerability_types and confidence > 0.5:
            vulnerability_types.append(VulnerabilityType.UNKNOWN)
        
        return vulnerability_types
    
    async def _analyze_vulnerabilities(self, 
                                     contract_code: str, 
                                     model_predictions: List[ModelPrediction]) -> List[VulnerabilityDetail]:
        """Analyze and create detailed vulnerability information"""
        vulnerabilities = []
        
        # Get all unique vulnerability types from predictions
        all_vuln_types = set()
        for pred in model_predictions:
            all_vuln_types.update(pred.vulnerability_types)
        
        # Create detailed vulnerability information
        for vuln_type in all_vuln_types:
            if vuln_type == VulnerabilityType.UNKNOWN:
                continue
            
            # Get pattern information
            pattern_info = self.vulnerability_patterns.get(vuln_type.value, {})
            
            # Calculate confidence based on model predictions
            confidence = max([pred.confidence for pred in model_predictions 
                            if vuln_type in pred.vulnerability_types], default=0.0)
            
            # Create vulnerability detail
            vuln_detail = VulnerabilityDetail(
                type=vuln_type,
                severity=pattern_info.get('severity', VulnerabilitySeverity.MEDIUM),
                confidence=confidence,
                description=self._get_vulnerability_description(vuln_type),
                location=self._find_vulnerability_location(contract_code, vuln_type),
                explanation=self._get_vulnerability_explanation(vuln_type),
                recommendation=self._get_vulnerability_recommendation(vuln_type),
                references=self._get_vulnerability_references(vuln_type)
            )
            
            vulnerabilities.append(vuln_detail)
        
        return vulnerabilities
    
    def _get_vulnerability_description(self, vuln_type: VulnerabilityType) -> str:
        """Get human-readable vulnerability description"""
        descriptions = {
            VulnerabilityType.REENTRANCY: "Reentrancy vulnerability allows external calls to re-enter the contract",
            VulnerabilityType.INTEGER_OVERFLOW: "Integer overflow/underflow can cause unexpected behavior",
            VulnerabilityType.ACCESS_CONTROL: "Improper access control allows unauthorized operations",
            VulnerabilityType.UNCHECKED_EXTERNAL_CALLS: "Unchecked external calls can fail silently",
            VulnerabilityType.FRONT_RUNNING: "Transaction ordering dependency allows front-running attacks",
            VulnerabilityType.TIMESTAMP_DEPENDENCE: "Block timestamp can be manipulated by miners",
            VulnerabilityType.GAS_LIMIT: "Gas limit issues can cause transaction failures",
            VulnerabilityType.DENIAL_OF_SERVICE: "Contract can be made to fail or consume excessive gas",
            VulnerabilityType.TX_ORIGIN: "Using tx.origin for authorization is vulnerable to phishing"
        }
        return descriptions.get(vuln_type, "Unknown vulnerability type")
    
    def _find_vulnerability_location(self, contract_code: str, vuln_type: VulnerabilityType) -> Optional[str]:
        """Find the location of a vulnerability in the code"""
        pattern_info = self.vulnerability_patterns.get(vuln_type.value, {})
        patterns = pattern_info.get('patterns', [])
        
        for i, line in enumerate(contract_code.split('\n')):
            for pattern in patterns:
                if pattern in line:
                    return f"Line {i+1}: {line.strip()}"
        
        return None
    
    def _get_vulnerability_explanation(self, vuln_type: VulnerabilityType) -> str:
        """Get detailed technical explanation"""
        explanations = {
            VulnerabilityType.REENTRANCY: "Reentrancy occurs when external calls are made before state changes, allowing attackers to re-enter the function and drain funds.",
            VulnerabilityType.INTEGER_OVERFLOW: "Integer overflow happens when arithmetic operations exceed the maximum value for the data type, causing wraparound behavior.",
            VulnerabilityType.ACCESS_CONTROL: "Access control vulnerabilities occur when functions lack proper authorization checks, allowing unauthorized users to execute privileged operations."
        }
        return explanations.get(vuln_type, "Technical details not available")
    
    def _get_vulnerability_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get fix recommendations"""
        recommendations = {
            VulnerabilityType.REENTRANCY: "Use the checks-effects-interactions pattern and consider reentrancy guards.",
            VulnerabilityType.INTEGER_OVERFLOW: "Use SafeMath library or Solidity 0.8+ built-in overflow protection.",
            VulnerabilityType.ACCESS_CONTROL: "Implement proper access control using modifiers and role-based permissions."
        }
        return recommendations.get(vuln_type, "Review the code and implement appropriate security measures")
    
    def _get_vulnerability_references(self, vuln_type: VulnerabilityType) -> List[str]:
        """Get external references"""
        references = {
            VulnerabilityType.REENTRANCY: [
                "https://consensys.github.io/smart-contract-best-practices/attacks/reentrancy/",
                "https://swcregistry.io/docs/SWC-107"
            ],
            VulnerabilityType.INTEGER_OVERFLOW: [
                "https://swcregistry.io/docs/SWC-101",
                "https://consensys.github.io/smart-contract-best-practices/development-recommendations/solidity-specific/integer-arithmetic/"
            ]
        }
        return references.get(vuln_type, [])
    
    async def _generate_optimization_suggestions(self, contract_code: str) -> List[OptimizationSuggestion]:
        """Generate gas optimization suggestions"""
        suggestions = []
        
        # Check for storage optimization opportunities
        if 'uint256' in contract_code and 'mapping' in contract_code:
            suggestions.append(OptimizationSuggestion(
                type="storage_optimization",
                description="Consider using smaller data types and packed structs to reduce gas costs",
                potential_savings="10-30%",
                implementation="Use uint128 instead of uint256 where possible, pack structs",
                priority="medium"
            ))
        
        # Check for loop optimization
        if 'for (' in contract_code or 'while (' in contract_code:
            suggestions.append(OptimizationSuggestion(
                type="loop_optimization",
                description="Consider loop unrolling and batch operations for gas efficiency",
                potential_savings="5-15%",
                implementation="Unroll small loops, use batch operations",
                priority="low"
            ))
        
        # Check for function optimization
        if 'function' in contract_code:
            suggestions.append(OptimizationSuggestion(
                type="function_optimization",
                description="Use view/pure functions when possible to reduce gas costs",
                potential_savings="5-10%",
                implementation="Mark functions as view/pure when they don't modify state",
                priority="low"
            ))
        
        return suggestions
    
    def _calculate_risk_score(self, vulnerabilities: List[VulnerabilityDetail]) -> float:
        """Calculate overall risk score (0-10)"""
        if not vulnerabilities:
            return 0.0
        
        # Weight by severity
        severity_weights = {
            VulnerabilitySeverity.LOW: 1.0,
            VulnerabilitySeverity.MEDIUM: 3.0,
            VulnerabilitySeverity.HIGH: 6.0,
            VulnerabilitySeverity.CRITICAL: 10.0
        }
        
        total_score = 0.0
        for vuln in vulnerabilities:
            weight = severity_weights.get(vuln.severity, 1.0)
            total_score += weight * vuln.confidence
        
        # Normalize to 0-10 scale
        max_possible_score = len(vulnerabilities) * 10.0
        return min(total_score / max_possible_score * 10.0, 10.0)

class InferenceEngine:
    """Main inference engine for vulnerability detection"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_manager = ModelManager(model_dir)
        self.analyzer = VulnerabilityAnalyzer(self.model_manager)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._initialized = False
    
    async def initialize(self):
        """Initialize the inference engine"""
        if self._initialized:
            return
        
        logger.info("Initializing inference engine...")
        
        # Load models
        codebert_loaded = self.model_manager.load_model(ModelType.CODEBERT)
        gnn_loaded = self.model_manager.load_model(ModelType.GNN)
        
        if not codebert_loaded and not gnn_loaded:
            logger.warning("No models loaded, will use rule-based analysis")
        
        self._initialized = True
        logger.info("Inference engine initialized successfully")
    
    async def analyze_contract(self, 
                             contract_code: str, 
                             model_type: ModelType = ModelType.ENSEMBLE,
                             contract_name: Optional[str] = None) -> VulnerabilityReport:
        """Analyze a smart contract for vulnerabilities"""
        if not self._initialized:
            await self.initialize()
        
        # Set contract name in report
        report = await self.analyzer.analyze_contract(contract_code, model_type)
        if contract_name:
            report.contract_name = contract_name
        
        return report
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return self.model_manager.get_model_info()
    
    def is_ready(self) -> bool:
        """Check if the inference engine is ready"""
        return self._initialized and (
            self.model_manager.is_model_loaded(ModelType.CODEBERT) or 
            self.model_manager.is_model_loaded(ModelType.GNN)
        )
    
    async def shutdown(self):
        """Shutdown the inference engine"""
        logger.info("Shutting down inference engine...")
        self.executor.shutdown(wait=True)
        logger.info("Inference engine shutdown complete")
