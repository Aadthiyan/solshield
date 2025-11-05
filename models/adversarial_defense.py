#!/usr/bin/env python3
"""
Adversarial Training and Defense for Smart Contract Vulnerability Detection

This module implements robust adversarial training and defense mechanisms
to protect the ML models against adversarial attacks. It includes:

1. Adversarial Sample Generation
2. Adversarial Training Pipeline
3. Defense Mechanisms
4. Robustness Evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import random
import re
import ast
from abc import ABC, abstractmethod

@dataclass
class AdversarialSample:
    """Represents an adversarial sample"""
    original_code: str
    adversarial_code: str
    perturbation_type: str
    perturbation_strength: float
    original_label: int
    adversarial_label: int
    success: bool
    confidence_drop: float

class AdversarialGenerator(ABC):
    """Abstract base class for adversarial sample generation"""
    
    @abstractmethod
    def generate(self, code: str, target_label: int = None) -> AdversarialSample:
        """Generate an adversarial sample"""
        pass

class CodeObfuscationAttack(AdversarialGenerator):
    """Generates adversarial samples through code obfuscation"""
    
    def __init__(self, obfuscation_strength: float = 0.5):
        self.obfuscation_strength = obfuscation_strength
        self.obfuscation_techniques = [
            self._rename_variables,
            self._add_dead_code,
            self._reorder_statements,
            self._add_whitespace,
            self._insert_comments
        ]
    
    def generate(self, code: str, target_label: int = None) -> AdversarialSample:
        """Generate obfuscated adversarial sample"""
        original_code = code
        adversarial_code = code
        
        # Apply random obfuscation techniques
        num_techniques = int(len(self.obfuscation_techniques) * self.obfuscation_strength)
        selected_techniques = random.sample(self.obfuscation_techniques, num_techniques)
        
        for technique in selected_techniques:
            adversarial_code = technique(adversarial_code)
        
        return AdversarialSample(
            original_code=original_code,
            adversarial_code=adversarial_code,
            perturbation_type="obfuscation",
            perturbation_strength=self.obfuscation_strength,
            original_label=0,  # Will be set by caller
            adversarial_label=0,  # Will be set by caller
            success=False,  # Will be determined by caller
            confidence_drop=0.0  # Will be calculated by caller
        )
    
    def _rename_variables(self, code: str) -> str:
        """Rename variables to obfuscate code"""
        # Find variable names
        variable_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
        variables = set(re.findall(variable_pattern, code))
        
        # Create mapping for renaming
        var_mapping = {}
        for var in variables:
            if len(var) > 2 and not var in ['function', 'contract', 'pragma', 'solidity', 'require', 'assert']:
                new_name = f"var_{len(var_mapping)}"
                var_mapping[var] = new_name
        
        # Apply renaming
        obfuscated_code = code
        for old_name, new_name in var_mapping.items():
            obfuscated_code = re.sub(r'\b' + old_name + r'\b', new_name, obfuscated_code)
        
        return obfuscated_code
    
    def _add_dead_code(self, code: str) -> str:
        """Add dead code that doesn't affect functionality"""
        dead_code = """
        // Dead code for obfuscation
        function unusedFunction() private pure returns (uint256) {
            uint256 temp = 0;
            for (uint256 i = 0; i < 10; i++) {
                temp += i;
            }
            return temp;
        }
        """
        
        # Insert dead code before the last closing brace
        if '}' in code:
            last_brace = code.rfind('}')
            code = code[:last_brace] + dead_code + '\n' + code[last_brace:]
        
        return code
    
    def _reorder_statements(self, code: str) -> str:
        """Reorder statements within functions"""
        # Simple reordering by swapping adjacent lines
        lines = code.split('\n')
        for i in range(0, len(lines) - 1, 2):
            if random.random() < 0.3:  # 30% chance to swap
                lines[i], lines[i + 1] = lines[i + 1], lines[i]
        
        return '\n'.join(lines)
    
    def _add_whitespace(self, code: str) -> str:
        """Add extra whitespace"""
        # Add random spaces and newlines
        lines = code.split('\n')
        modified_lines = []
        
        for line in lines:
            if random.random() < 0.2:  # 20% chance to add whitespace
                modified_lines.append(' ' * random.randint(1, 4) + line)
            else:
                modified_lines.append(line)
        
        return '\n'.join(modified_lines)
    
    def _insert_comments(self, code: str) -> str:
        """Insert random comments"""
        lines = code.split('\n')
        modified_lines = []
        
        for line in lines:
            modified_lines.append(line)
            if random.random() < 0.1:  # 10% chance to add comment
                comment = f"// Random comment {random.randint(1, 100)}"
                modified_lines.append(comment)
        
        return '\n'.join(modified_lines)

class SemanticPerturbationAttack(AdversarialGenerator):
    """Generates adversarial samples through semantic perturbations"""
    
    def __init__(self, perturbation_strength: float = 0.3):
        self.perturbation_strength = perturbation_strength
        self.semantic_transformations = [
            self._add_redundant_checks,
            self._modify_conditions,
            self._add_intermediate_variables,
            self._change_operator_precedence
        ]
    
    def generate(self, code: str, target_label: int = None) -> AdversarialSample:
        """Generate semantically perturbed adversarial sample"""
        original_code = code
        adversarial_code = code
        
        # Apply semantic transformations
        for transformation in self.semantic_transformations:
            if random.random() < self.perturbation_strength:
                adversarial_code = transformation(adversarial_code)
        
        return AdversarialSample(
            original_code=original_code,
            adversarial_code=adversarial_code,
            perturbation_type="semantic",
            perturbation_strength=self.perturbation_strength,
            original_label=0,
            adversarial_label=0,
            success=False,
            confidence_drop=0.0
        )
    
    def _add_redundant_checks(self, code: str) -> str:
        """Add redundant require statements"""
        # Find existing require statements and add redundant ones
        require_pattern = r'require\s*\([^)]+\)'
        requires = re.findall(require_pattern, code)
        
        if requires:
            # Add a redundant check
            redundant_check = "require(true, \"Redundant check\");"
            code = code.replace(requires[0], requires[0] + '\n        ' + redundant_check)
        
        return code
    
    def _modify_conditions(self, code: str) -> str:
        """Modify conditions to be more complex"""
        # Find simple conditions and make them more complex
        condition_pattern = r'require\s*\(\s*(\w+)\s*\)'
        matches = re.findall(condition_pattern, code)
        
        for match in matches:
            complex_condition = f"{match} && true"
            code = code.replace(f"require({match})", f"require({complex_condition})")
        
        return code
    
    def _add_intermediate_variables(self, code: str) -> str:
        """Add intermediate variables"""
        # Find simple expressions and add intermediate variables
        assignment_pattern = r'(\w+)\s*=\s*([^;]+);'
        matches = re.findall(assignment_pattern, code)
        
        for var, expr in matches:
            if len(expr) > 10:  # Only for complex expressions
                intermediate_var = f"temp_{random.randint(1, 100)}"
                code = code.replace(f"{var} = {expr};", f"uint256 {intermediate_var} = {expr};\n        {var} = {intermediate_var};")
        
        return code
    
    def _change_operator_precedence(self, code: str) -> str:
        """Change operator precedence with parentheses"""
        # Add unnecessary parentheses
        arithmetic_pattern = r'(\w+)\s*([+\-*/])\s*(\w+)'
        matches = re.findall(arithmetic_pattern, code)
        
        for left, op, right in matches:
            if random.random() < 0.3:
                new_expr = f"({left} {op} {right})"
                code = code.replace(f"{left} {op} {right}", new_expr)
        
        return code

class GradientBasedAttack(AdversarialGenerator):
    """Generates adversarial samples using gradient-based methods"""
    
    def __init__(self, epsilon: float = 0.1, num_iterations: int = 10):
        self.epsilon = epsilon
        self.num_iterations = num_iterations
    
    def generate(self, code: str, target_label: int = None) -> AdversarialSample:
        """Generate gradient-based adversarial sample"""
        # This is a simplified version - in practice, you'd need to work with
        # the actual model embeddings and gradients
        original_code = code
        
        # Simulate gradient-based perturbation
        adversarial_code = self._simulate_gradient_attack(code)
        
        return AdversarialSample(
            original_code=original_code,
            adversarial_code=adversarial_code,
            perturbation_type="gradient",
            perturbation_strength=self.epsilon,
            original_label=0,
            adversarial_label=0,
            success=False,
            confidence_drop=0.0
        )
    
    def _simulate_gradient_attack(self, code: str) -> str:
        """Simulate gradient-based attack on code"""
        # In practice, this would use actual gradients from the model
        # For now, we'll simulate with random perturbations
        
        lines = code.split('\n')
        modified_lines = []
        
        for line in lines:
            if random.random() < self.epsilon:
                # Simulate gradient-based modification
                if 'require' in line:
                    line = line.replace('require', 'require')  # Simulate perturbation
            modified_lines.append(line)
        
        return '\n'.join(modified_lines)

class AdversarialTrainingPipeline:
    """Pipeline for adversarial training"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.adversarial_generators = [
            CodeObfuscationAttack(),
            SemanticPerturbationAttack(),
            GradientBasedAttack()
        ]
    
    def adversarial_training_step(self, batch: Dict[str, Any], 
                                optimizer: optim.Optimizer) -> Dict[str, float]:
        """Perform one step of adversarial training"""
        self.model.train()
        
        # Get original predictions
        original_outputs = self.model(batch['x'], batch['edge_index'], 
                                   batch.get('edge_attr', None), batch['batch'])
        original_loss = F.binary_cross_entropy(original_outputs['vulnerability_prediction'], 
                                              batch['y'].unsqueeze(1))
        
        # Generate adversarial samples
        adversarial_samples = []
        for i in range(len(batch['x'])):
            # This is simplified - in practice, you'd need to convert tensors back to code
            adversarial_sample = self._generate_adversarial_sample(batch, i)
            adversarial_samples.append(adversarial_sample)
        
        # Train on adversarial samples
        adversarial_loss = 0.0
        for sample in adversarial_samples:
            if sample.success:  # Only use successful adversarial samples
                # Convert adversarial code back to tensor format
                adv_batch = self._convert_adversarial_to_batch(sample, batch)
                adv_outputs = self.model(adv_batch['x'], adv_batch['edge_index'], 
                                       adv_batch.get('edge_attr', None), adv_batch['batch'])
                adversarial_loss += F.binary_cross_entropy(adv_outputs['vulnerability_prediction'], 
                                                        adv_batch['y'].unsqueeze(1))
        
        # Combined loss
        total_loss = original_loss + 0.5 * adversarial_loss
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return {
            'original_loss': original_loss.item(),
            'adversarial_loss': adversarial_loss.item() if adversarial_loss > 0 else 0.0,
            'total_loss': total_loss.item()
        }
    
    def _generate_adversarial_sample(self, batch: Dict[str, Any], index: int) -> AdversarialSample:
        """Generate adversarial sample for a specific item in the batch"""
        # Select random generator
        generator = random.choice(self.adversarial_generators)
        
        # This is simplified - in practice, you'd need to convert tensor to code
        # and back to tensor
        code = f"contract_{index}"  # Placeholder
        adversarial_sample = generator.generate(code)
        
        # Simulate success/failure
        adversarial_sample.success = random.random() < 0.7  # 70% success rate
        
        return adversarial_sample
    
    def _convert_adversarial_to_batch(self, sample: AdversarialSample, 
                                    original_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Convert adversarial sample back to batch format"""
        # This is simplified - in practice, you'd need to convert code back to tensors
        return original_batch

class DefenseMechanisms:
    """Implements various defense mechanisms against adversarial attacks"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.detection_threshold = 0.5
        self.robustness_threshold = 0.8
    
    def adversarial_detection(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Detect potential adversarial samples"""
        # Use attention weights and confidence scores for detection
        attention_weights = outputs.get('attention_weights', None)
        vulnerability_pred = outputs.get('vulnerability_prediction', None)
        
        if attention_weights is not None and vulnerability_pred is not None:
            # Calculate attention entropy (high entropy might indicate adversarial)
            attention_entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
            
            # Calculate prediction confidence
            confidence = torch.abs(vulnerability_pred - 0.5) * 2  # Convert to 0-1 scale
            
            # Combine signals for detection
            detection_score = attention_entropy * (1 - confidence)
            is_adversarial = detection_score > self.detection_threshold
            
            return is_adversarial
        
        return torch.zeros(vulnerability_pred.shape[0], dtype=torch.bool)
    
    def input_sanitization(self, code: str) -> str:
        """Sanitize input to remove potential adversarial patterns"""
        # Remove suspicious patterns
        suspicious_patterns = [
            r'//.*?$',  # Comments
            r'/\*.*?\*/',  # Multi-line comments
            r'\s+',  # Multiple whitespace
        ]
        
        sanitized_code = code
        for pattern in suspicious_patterns:
            sanitized_code = re.sub(pattern, ' ', sanitized_code, flags=re.MULTILINE | re.DOTALL)
        
        # Normalize whitespace
        sanitized_code = re.sub(r' +', ' ', sanitized_code)
        sanitized_code = re.sub(r'\n\s*\n', '\n', sanitized_code)
        
        return sanitized_code.strip()
    
    def robust_inference(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Perform robust inference with defense mechanisms"""
        # Sanitize inputs
        sanitized_batch = self._sanitize_batch(batch)
        
        # Get model outputs
        outputs = self.model(sanitized_batch['x'], sanitized_batch['edge_index'], 
                           sanitized_batch.get('edge_attr', None), sanitized_batch['batch'])
        
        # Apply adversarial detection
        is_adversarial = self.adversarial_detection(outputs)
        
        # Adjust predictions for detected adversarial samples
        if is_adversarial.any():
            # Reduce confidence for adversarial samples
            vulnerability_pred = outputs['vulnerability_prediction']
            vulnerability_pred[is_adversarial] = 0.5  # Neutral prediction
            outputs['vulnerability_prediction'] = vulnerability_pred
        
        return outputs
    
    def _sanitize_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize batch data"""
        # This is simplified - in practice, you'd need to sanitize the actual code
        return batch

class RobustnessEvaluator:
    """Evaluates model robustness against adversarial attacks"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.defense_mechanisms = DefenseMechanisms(model)
        self.adversarial_generators = [
            CodeObfuscationAttack(),
            SemanticPerturbationAttack(),
            GradientBasedAttack()
        ]
    
    def evaluate_robustness(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate model robustness against various attacks"""
        self.model.eval()
        
        results = {
            'original_accuracy': 0.0,
            'obfuscation_robustness': 0.0,
            'semantic_robustness': 0.0,
            'gradient_robustness': 0.0,
            'overall_robustness': 0.0,
            'adversarial_detection_rate': 0.0
        }
        
        total_samples = 0
        correct_original = 0
        correct_adversarial = 0
        detected_adversarial = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Original accuracy
                original_outputs = self.model(batch['x'], batch['edge_index'], 
                                           batch.get('edge_attr', None), batch['batch'])
                original_pred = (original_outputs['vulnerability_prediction'] > 0.5).float()
                correct_original += (original_pred.squeeze() == batch['y']).sum().item()
                
                # Test against adversarial attacks
                for generator in self.adversarial_generators:
                    # Generate adversarial samples (simplified)
                    adversarial_batch = self._generate_adversarial_batch(batch, generator)
                    
                    # Test with defense mechanisms
                    robust_outputs = self.defense_mechanisms.robust_inference(adversarial_batch)
                    
                    # Check if adversarial samples are detected
                    is_adversarial = self.defense_mechanisms.adversarial_detection(robust_outputs)
                    detected_adversarial += is_adversarial.sum().item()
                    
                    # Calculate accuracy on adversarial samples
                    adv_pred = (robust_outputs['vulnerability_prediction'] > 0.5).float()
                    correct_adversarial += (adv_pred.squeeze() == batch['y']).sum().item()
                
                total_samples += batch['y'].shape[0]
        
        # Calculate metrics
        results['original_accuracy'] = correct_original / total_samples
        results['adversarial_accuracy'] = correct_adversarial / (total_samples * len(self.adversarial_generators))
        results['adversarial_detection_rate'] = detected_adversarial / (total_samples * len(self.adversarial_generators))
        results['overall_robustness'] = (results['original_accuracy'] + results['adversarial_accuracy']) / 2
        
        return results
    
    def _generate_adversarial_batch(self, batch: Dict[str, Any], 
                                  generator: AdversarialGenerator) -> Dict[str, Any]:
        """Generate adversarial batch using specified generator"""
        # This is simplified - in practice, you'd convert tensors to code,
        # apply adversarial transformations, and convert back to tensors
        return batch

# Example usage and testing
if __name__ == "__main__":
    # Test adversarial generation
    sample_contract = """
    pragma solidity ^0.8.0;
    
    contract TestContract {
        mapping(address => uint256) public balances;
        
        function withdraw(uint256 amount) public {
            require(balances[msg.sender] >= amount, "Insufficient balance");
            (bool success, ) = msg.sender.call{value: amount}("");
            require(success, "Transfer failed");
            balances[msg.sender] -= amount;
        }
    }
    """
    
    # Test obfuscation attack
    obfuscation_attack = CodeObfuscationAttack()
    adversarial_sample = obfuscation_attack.generate(sample_contract)
    
    print("Original Code:")
    print(adversarial_sample.original_code[:100] + "...")
    print("\nAdversarial Code:")
    print(adversarial_sample.adversarial_code[:100] + "...")
    print(f"\nPerturbation Type: {adversarial_sample.perturbation_type}")
    print(f"Perturbation Strength: {adversarial_sample.perturbation_strength}")
    
    # Test semantic perturbation
    semantic_attack = SemanticPerturbationAttack()
    semantic_sample = semantic_attack.generate(sample_contract)
    
    print("\nSemantic Perturbation:")
    print(semantic_sample.adversarial_code[:100] + "...")
    
    # Test defense mechanisms
    defense = DefenseMechanisms(None)  # Would use actual model
    sanitized_code = defense.input_sanitization(sample_contract)
    print(f"\nSanitized Code Length: {len(sanitized_code)}")
