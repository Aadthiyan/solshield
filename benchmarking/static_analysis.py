"""
Static Analysis Tools Integration Module

This module provides integration with static analysis tools like Slither and Mythril
for benchmarking AI audit results against established security analysis tools.
"""

import subprocess
import json
import os
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class VulnerabilityFinding:
    """Represents a vulnerability finding from static analysis tools."""
    tool: str
    vulnerability_type: str
    severity: str
    confidence: float
    description: str
    location: str
    line_number: Optional[int] = None
    function_name: Optional[str] = None
    code_snippet: Optional[str] = None
    recommendation: Optional[str] = None


@dataclass
class AnalysisResult:
    """Represents the complete analysis result from a tool."""
    tool: str
    contract_path: str
    vulnerabilities: List[VulnerabilityFinding]
    analysis_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class SlitherAnalyzer:
    """Slither static analysis tool integration."""
    
    def __init__(self, slither_path: str = "slither"):
        self.slither_path = slither_path
        self.supported_formats = [".sol"]
    
    def analyze_contract(self, contract_path: str) -> AnalysisResult:
        """
        Analyze a smart contract using Slither.
        
        Args:
            contract_path: Path to the Solidity contract file
            
        Returns:
            AnalysisResult containing vulnerabilities and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Run Slither analysis
            cmd = [
                self.slither_path,
                contract_path,
                "--json",
                "--disable-color",
                "--exclude-dependencies"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            analysis_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"Slither analysis failed: {result.stderr}")
                return AnalysisResult(
                    tool="slither",
                    contract_path=contract_path,
                    vulnerabilities=[],
                    analysis_time=analysis_time,
                    success=False,
                    error_message=result.stderr
                )
            
            # Parse JSON output
            vulnerabilities = self._parse_slither_output(result.stdout)
            
            return AnalysisResult(
                tool="slither",
                contract_path=contract_path,
                vulnerabilities=vulnerabilities,
                analysis_time=analysis_time,
                success=True,
                metadata={"stdout": result.stdout}
            )
            
        except subprocess.TimeoutExpired:
            logger.error(f"Slither analysis timed out for {contract_path}")
            return AnalysisResult(
                tool="slither",
                contract_path=contract_path,
                vulnerabilities=[],
                analysis_time=time.time() - start_time,
                success=False,
                error_message="Analysis timed out"
            )
        except Exception as e:
            logger.error(f"Slither analysis error: {str(e)}")
            return AnalysisResult(
                tool="slither",
                contract_path=contract_path,
                vulnerabilities=[],
                analysis_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _parse_slither_output(self, output: str) -> List[VulnerabilityFinding]:
        """Parse Slither JSON output into VulnerabilityFinding objects."""
        vulnerabilities = []
        
        try:
            data = json.loads(output)
            
            for detector_name, findings in data.get("results", {}).get("detectors", {}).items():
                for finding in findings:
                    vulnerability = VulnerabilityFinding(
                        tool="slither",
                        vulnerability_type=detector_name,
                        severity=self._map_slither_severity(finding.get("impact", "unknown")),
                        confidence=finding.get("confidence", 0.0) / 100.0,  # Convert to 0-1 scale
                        description=finding.get("description", ""),
                        location=finding.get("elements", [{}])[0].get("source_mapping", {}).get("filename_short", ""),
                        line_number=finding.get("elements", [{}])[0].get("source_mapping", {}).get("lines", [None])[0],
                        function_name=finding.get("elements", [{}])[0].get("name", ""),
                        code_snippet=finding.get("elements", [{}])[0].get("source_mapping", {}).get("content", ""),
                        recommendation=finding.get("description", "")
                    )
                    vulnerabilities.append(vulnerability)
                    
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Slither JSON output: {e}")
        
        return vulnerabilities
    
    def _map_slither_severity(self, impact: str) -> str:
        """Map Slither impact levels to standard severity levels."""
        severity_mapping = {
            "High": "high",
            "Medium": "medium", 
            "Low": "low",
            "Informational": "low"
        }
        return severity_mapping.get(impact, "unknown")


class MythrilAnalyzer:
    """Mythril static analysis tool integration."""
    
    def __init__(self, myth_path: str = "myth"):
        self.myth_path = myth_path
        self.supported_formats = [".sol"]
    
    def analyze_contract(self, contract_path: str) -> AnalysisResult:
        """
        Analyze a smart contract using Mythril.
        
        Args:
            contract_path: Path to the Solidity contract file
            
        Returns:
            AnalysisResult containing vulnerabilities and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Run Mythril analysis
            cmd = [
                self.myth_path,
                "analyze",
                contract_path,
                "--execution-timeout", "300",
                "--solver-timeout", "60",
                "--max-depth", "50",
                "--output", "json"
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            analysis_time = time.time() - start_time
            
            if result.returncode != 0:
                logger.error(f"Mythril analysis failed: {result.stderr}")
                return AnalysisResult(
                    tool="mythril",
                    contract_path=contract_path,
                    vulnerabilities=[],
                    analysis_time=analysis_time,
                    success=False,
                    error_message=result.stderr
                )
            
            # Parse JSON output
            vulnerabilities = self._parse_mythril_output(result.stdout)
            
            return AnalysisResult(
                tool="mythril",
                contract_path=contract_path,
                vulnerabilities=vulnerabilities,
                analysis_time=analysis_time,
                success=True,
                metadata={"stdout": result.stdout}
            )
            
        except subprocess.TimeoutExpired:
            logger.error(f"Mythril analysis timed out for {contract_path}")
            return AnalysisResult(
                tool="mythril",
                contract_path=contract_path,
                vulnerabilities=[],
                analysis_time=time.time() - start_time,
                success=False,
                error_message="Analysis timed out"
            )
        except Exception as e:
            logger.error(f"Mythril analysis error: {str(e)}")
            return AnalysisResult(
                tool="mythril",
                contract_path=contract_path,
                vulnerabilities=[],
                analysis_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _parse_mythril_output(self, output: str) -> List[VulnerabilityFinding]:
        """Parse Mythril JSON output into VulnerabilityFinding objects."""
        vulnerabilities = []
        
        try:
            data = json.loads(output)
            
            for issue in data.get("issues", []):
                vulnerability = VulnerabilityFinding(
                    tool="mythril",
                    vulnerability_type=issue.get("title", "unknown"),
                    severity=self._map_mythril_severity(issue.get("severity", "unknown")),
                    confidence=issue.get("confidence", 0.0) / 100.0,  # Convert to 0-1 scale
                    description=issue.get("description", ""),
                    location=issue.get("filename", ""),
                    line_number=issue.get("lineno"),
                    function_name=issue.get("function"),
                    code_snippet=issue.get("code"),
                    recommendation=issue.get("description", "")
                )
                vulnerabilities.append(vulnerability)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Mythril JSON output: {e}")
        
        return vulnerabilities
    
    def _map_mythril_severity(self, severity: str) -> str:
        """Map Mythril severity levels to standard severity levels."""
        severity_mapping = {
            "High": "high",
            "Medium": "medium",
            "Low": "low",
            "Informational": "low"
        }
        return severity_mapping.get(severity, "unknown")


class StaticAnalysisBenchmark:
    """Benchmark AI audit results against static analysis tools."""
    
    def __init__(self):
        self.slither = SlitherAnalyzer()
        self.mythril = MythrilAnalyzer()
        self.tools = [self.slither, self.mythril]
    
    def benchmark_contract(self, contract_path: str, ai_results: List[VulnerabilityFinding]) -> Dict[str, Any]:
        """
        Benchmark AI results against static analysis tools for a single contract.
        
        Args:
            contract_path: Path to the contract file
            ai_results: AI analysis results
            
        Returns:
            Benchmark results dictionary
        """
        benchmark_results = {
            "contract_path": contract_path,
            "ai_results": ai_results,
            "static_analysis_results": {},
            "comparison": {},
            "summary": {}
        }
        
        # Run static analysis tools
        for tool in self.tools:
            logger.info(f"Running {tool.__class__.__name__} analysis on {contract_path}")
            result = tool.analyze_contract(contract_path)
            benchmark_results["static_analysis_results"][result.tool] = result
        
        # Compare results
        benchmark_results["comparison"] = self._compare_results(
            ai_results, 
            benchmark_results["static_analysis_results"]
        )
        
        # Generate summary
        benchmark_results["summary"] = self._generate_summary(benchmark_results)
        
        return benchmark_results
    
    def _compare_results(self, ai_results: List[VulnerabilityFinding], 
                        static_results: Dict[str, AnalysisResult]) -> Dict[str, Any]:
        """Compare AI results with static analysis results."""
        comparison = {
            "precision": {},
            "recall": {},
            "f1_score": {},
            "overlap_analysis": {},
            "unique_findings": {}
        }
        
        # Calculate metrics for each static analysis tool
        for tool_name, static_result in static_results.items():
            if not static_result.success:
                continue
                
            static_vulns = static_result.vulnerabilities
            
            # Calculate precision, recall, and F1 score
            precision, recall, f1 = self._calculate_metrics(ai_results, static_vulns)
            
            comparison["precision"][tool_name] = precision
            comparison["recall"][tool_name] = recall
            comparison["f1_score"][tool_name] = f1
            
            # Analyze overlaps and unique findings
            overlap_analysis = self._analyze_overlaps(ai_results, static_vulns)
            comparison["overlap_analysis"][tool_name] = overlap_analysis
            
            # Find unique findings
            unique_ai = self._find_unique_vulnerabilities(ai_results, static_vulns)
            unique_static = self._find_unique_vulnerabilities(static_vulns, ai_results)
            
            comparison["unique_findings"][tool_name] = {
                "ai_unique": unique_ai,
                "static_unique": unique_static
            }
        
        return comparison
    
    def _calculate_metrics(self, ai_results: List[VulnerabilityFinding], 
                          static_results: List[VulnerabilityFinding]) -> Tuple[float, float, float]:
        """Calculate precision, recall, and F1 score."""
        if not ai_results and not static_results:
            return 1.0, 1.0, 1.0
        
        if not ai_results:
            return 0.0, 0.0, 0.0
        
        if not static_results:
            return 0.0, 0.0, 0.0
        
        # Simple overlap calculation based on vulnerability type and location
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        ai_matched = set()
        static_matched = set()
        
        for i, ai_vuln in enumerate(ai_results):
            matched = False
            for j, static_vuln in enumerate(static_results):
                if self._vulnerabilities_match(ai_vuln, static_vuln):
                    true_positives += 1
                    ai_matched.add(i)
                    static_matched.add(j)
                    matched = True
                    break
            
            if not matched:
                false_positives += 1
        
        false_negatives = len(static_results) - len(static_matched)
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _vulnerabilities_match(self, vuln1: VulnerabilityFinding, vuln2: VulnerabilityFinding) -> bool:
        """Check if two vulnerabilities represent the same issue."""
        # Match based on vulnerability type and location
        type_match = vuln1.vulnerability_type.lower() == vuln2.vulnerability_type.lower()
        location_match = vuln1.location == vuln2.location
        
        # Consider line numbers if available
        line_match = True
        if vuln1.line_number and vuln2.line_number:
            line_match = abs(vuln1.line_number - vuln2.line_number) <= 5  # Allow 5 line tolerance
        
        return type_match and location_match and line_match
    
    def _analyze_overlaps(self, ai_results: List[VulnerabilityFinding], 
                         static_results: List[VulnerabilityFinding]) -> Dict[str, Any]:
        """Analyze overlaps between AI and static analysis results."""
        overlaps = []
        
        for ai_vuln in ai_results:
            for static_vuln in static_results:
                if self._vulnerabilities_match(ai_vuln, static_vuln):
                    overlaps.append({
                        "ai_vulnerability": ai_vuln,
                        "static_vulnerability": static_vuln,
                        "severity_match": ai_vuln.severity == static_vuln.severity,
                        "confidence_diff": abs(ai_vuln.confidence - static_vuln.confidence)
                    })
        
        return {
            "total_overlaps": len(overlaps),
            "overlap_details": overlaps,
            "overlap_percentage": len(overlaps) / max(len(ai_results), len(static_results)) * 100 if max(len(ai_results), len(static_results)) > 0 else 0
        }
    
    def _find_unique_vulnerabilities(self, vulns1: List[VulnerabilityFinding], 
                                   vulns2: List[VulnerabilityFinding]) -> List[VulnerabilityFinding]:
        """Find vulnerabilities that are unique to the first list."""
        unique = []
        
        for vuln1 in vulns1:
            is_unique = True
            for vuln2 in vulns2:
                if self._vulnerabilities_match(vuln1, vuln2):
                    is_unique = False
                    break
            
            if is_unique:
                unique.append(vuln1)
        
        return unique
    
    def _generate_summary(self, benchmark_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of benchmark results."""
        summary = {
            "total_contracts": 1,
            "ai_total_findings": len(benchmark_results["ai_results"]),
            "static_analysis_summary": {},
            "overall_performance": {}
        }
        
        # Summarize static analysis results
        for tool_name, result in benchmark_results["static_analysis_results"].items():
            summary["static_analysis_summary"][tool_name] = {
                "success": result.success,
                "total_findings": len(result.vulnerabilities),
                "analysis_time": result.analysis_time,
                "error_message": result.error_message
            }
        
        # Calculate overall performance metrics
        comparison = benchmark_results["comparison"]
        if comparison.get("precision") and comparison.get("recall") and comparison.get("f1_score"):
            avg_precision = sum(comparison["precision"].values()) / len(comparison["precision"])
            avg_recall = sum(comparison["recall"].values()) / len(comparison["recall"])
            avg_f1 = sum(comparison["f1_score"].values()) / len(comparison["f1_score"])
            
            summary["overall_performance"] = {
                "average_precision": avg_precision,
                "average_recall": avg_recall,
                "average_f1_score": avg_f1
            }
        
        return summary


def create_vulnerability_finding_from_ai(ai_result: Dict[str, Any]) -> VulnerabilityFinding:
    """Convert AI analysis result to VulnerabilityFinding object."""
    return VulnerabilityFinding(
        tool="ai_model",
        vulnerability_type=ai_result.get("type", "unknown"),
        severity=ai_result.get("severity", "unknown"),
        confidence=ai_result.get("confidence", 0.0),
        description=ai_result.get("description", ""),
        location=ai_result.get("location", ""),
        line_number=ai_result.get("line_number"),
        function_name=ai_result.get("function_name"),
        code_snippet=ai_result.get("code_snippet"),
        recommendation=ai_result.get("recommendation")
    )
