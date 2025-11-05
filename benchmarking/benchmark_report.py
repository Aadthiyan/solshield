"""
Benchmark Report Generator

This module generates comprehensive benchmark reports comparing AI vulnerability
detection against static analysis tools like Slither and Mythril.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Result from a static analysis tool."""
    tool_name: str
    vulnerabilities_found: List[Dict[str, Any]]
    execution_time: float
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    true_negatives: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0


@dataclass
class BenchmarkResult:
    """Result of benchmarking a single contract."""
    contract_id: str
    contract_name: str
    contract_code: str
    expected_vulnerabilities: List[Dict[str, Any]]
    ai_results: Dict[str, Any]
    static_analysis_results: Dict[str, ToolResult]
    overall_accuracy: float
    best_tool: str
    performance_metrics: Dict[str, Any]


class BenchmarkReportGenerator:
    """Generates comprehensive benchmark reports."""
    
    def __init__(self, output_dir: str = "benchmarking/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load benchmark data
        self.benchmark_data = self._load_benchmark_data()
        self.results = []
    
    def _load_benchmark_data(self) -> Dict[str, Any]:
        """Load benchmark dataset."""
        benchmark_path = Path("data/benchmarks/comprehensive_benchmark.json")
        
        if not benchmark_path.exists():
            # Create comprehensive benchmark dataset
            sample_data = {
                "contracts": [
                    {
                        "id": "reentrancy_vulnerable",
                        "name": "Reentrancy Vulnerable Contract",
                        "code": """
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
                        "expected_vulnerabilities": [
                            {
                                "type": "reentrancy",
                                "severity": "high",
                                "location": "withdraw function",
                                "description": "External call before state update",
                                "confidence": 0.9
                            }
                        ]
                    },
                    {
                        "id": "integer_overflow",
                        "name": "Integer Overflow Contract",
                        "code": """
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
                        "expected_vulnerabilities": [
                            {
                                "type": "integer_overflow",
                                "severity": "medium",
                                "location": "mint function",
                                "description": "Potential integer overflow",
                                "confidence": 0.8
                            }
                        ]
                    },
                    {
                        "id": "unchecked_call",
                        "name": "Unchecked Call Contract",
                        "code": """
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
                        "expected_vulnerabilities": [
                            {
                                "type": "unchecked_call",
                                "severity": "medium",
                                "location": "transfer function",
                                "description": "Unchecked external call",
                                "confidence": 0.7
                            }
                        ]
                    },
                    {
                        "id": "safe_contract",
                        "name": "Safe Contract",
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
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
}
""",
                        "expected_vulnerabilities": []
                    }
                ]
            }
            
            # Save sample data
            benchmark_path.parent.mkdir(parents=True, exist_ok=True)
            with open(benchmark_path, 'w') as f:
                json.dump(sample_data, f, indent=2)
            
            logger.info(f"Created comprehensive benchmark dataset at {benchmark_path}")
        
        with open(benchmark_path, 'r') as f:
            return json.load(f)
    
    def run_benchmark(self, api_endpoint: str = "http://localhost:8000") -> List[BenchmarkResult]:
        """Run comprehensive benchmark comparing AI vs static analysis tools."""
        logger.info("Starting comprehensive benchmark...")
        
        results = []
        
        for contract_data in self.benchmark_data["contracts"]:
            logger.info(f"Benchmarking contract: {contract_data['name']}")
            
            # Run AI analysis
            ai_results = self._run_ai_analysis(contract_data["code"], api_endpoint)
            
            # Run static analysis tools
            static_results = self._run_static_analysis(contract_data["code"])
            
            # Calculate metrics
            metrics = self._calculate_metrics(
                contract_data["expected_vulnerabilities"],
                ai_results,
                static_results
            )
            
            # Determine best tool
            best_tool = self._determine_best_tool(metrics)
            
            result = BenchmarkResult(
                contract_id=contract_data["id"],
                contract_name=contract_data["name"],
                contract_code=contract_data["code"],
                expected_vulnerabilities=contract_data["expected_vulnerabilities"],
                ai_results=ai_results,
                static_analysis_results=static_results,
                overall_accuracy=metrics["ai"]["accuracy"],
                best_tool=best_tool,
                performance_metrics=metrics
            )
            
            results.append(result)
            self.results.append(result)
        
        logger.info(f"Benchmark completed for {len(results)} contracts")
        return results
    
    def _run_ai_analysis(self, contract_code: str, api_endpoint: str) -> Dict[str, Any]:
        """Run AI vulnerability analysis."""
        try:
            import requests
            
            response = requests.post(
                f"{api_endpoint}/analyze",
                json={"contract_code": contract_code},
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"AI analysis failed: {response.status_code}")
                return {"vulnerabilities": [], "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            logger.error(f"AI analysis error: {e}")
            return {"vulnerabilities": [], "error": str(e)}
    
    def _run_static_analysis(self, contract_code: str) -> Dict[str, ToolResult]:
        """Run static analysis tools."""
        results = {}
        
        # Run Slither
        try:
            slither_result = self._run_slither(contract_code)
            results["slither"] = slither_result
        except Exception as e:
            logger.error(f"Slither analysis failed: {e}")
            results["slither"] = ToolResult(
                tool_name="slither",
                vulnerabilities_found=[],
                execution_time=0,
                false_positives=0,
                false_negatives=0,
                true_positives=0,
                true_negatives=0
            )
        
        # Run Mythril
        try:
            mythril_result = self._run_mythril(contract_code)
            results["mythril"] = mythril_result
        except Exception as e:
            logger.error(f"Mythril analysis failed: {e}")
            results["mythril"] = ToolResult(
                tool_name="mythril",
                vulnerabilities_found=[],
                execution_time=0,
                false_positives=0,
                false_negatives=0,
                true_positives=0,
                true_negatives=0
            )
        
        return results
    
    def _run_slither(self, contract_code: str) -> ToolResult:
        """Run Slither static analysis."""
        import subprocess
        import tempfile
        import time
        
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
            f.write(contract_code)
            temp_file = f.name
        
        try:
            # Run Slither
            result = subprocess.run(
                ["slither", temp_file, "--json", "-"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            vulnerabilities = []
            if result.returncode == 0 and result.stdout:
                try:
                    slither_output = json.loads(result.stdout)
                    vulnerabilities = self._parse_slither_output(slither_output)
                except json.JSONDecodeError:
                    logger.warning("Failed to parse Slither JSON output")
            
            return ToolResult(
                tool_name="slither",
                vulnerabilities_found=vulnerabilities,
                execution_time=execution_time
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def _run_mythril(self, contract_code: str) -> ToolResult:
        """Run Mythril static analysis."""
        import subprocess
        import tempfile
        import time
        
        start_time = time.time()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sol', delete=False) as f:
            f.write(contract_code)
            temp_file = f.name
        
        try:
            # Run Mythril
            result = subprocess.run(
                ["myth", "analyze", temp_file, "--execution-timeout", "60"],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            vulnerabilities = []
            if result.stdout:
                vulnerabilities = self._parse_mythril_output(result.stdout)
            
            return ToolResult(
                tool_name="mythril",
                vulnerabilities_found=vulnerabilities,
                execution_time=execution_time
            )
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    def _parse_slither_output(self, slither_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse Slither output."""
        vulnerabilities = []
        
        if "results" in slither_output:
            for detector in slither_output["results"]["detectors"]:
                vulnerabilities.append({
                    "type": detector.get("check", "unknown"),
                    "severity": detector.get("impact", "unknown"),
                    "description": detector.get("description", ""),
                    "location": detector.get("elements", [{}])[0].get("name", "unknown"),
                    "confidence": 0.8  # Default confidence for Slither
                })
        
        return vulnerabilities
    
    def _parse_mythril_output(self, mythril_output: str) -> List[Dict[str, Any]]:
        """Parse Mythril output."""
        vulnerabilities = []
        
        # Simple parsing - in practice, you'd want more robust parsing
        lines = mythril_output.split('\n')
        current_vuln = {}
        
        for line in lines:
            if "Vulnerability:" in line:
                if current_vuln:
                    vulnerabilities.append(current_vuln)
                current_vuln = {
                    "type": line.split("Vulnerability:")[1].strip(),
                    "severity": "medium",  # Default
                    "description": "",
                    "location": "unknown",
                    "confidence": 0.7  # Default confidence for Mythril
                }
            elif "Description:" in line and current_vuln:
                current_vuln["description"] = line.split("Description:")[1].strip()
        
        if current_vuln:
            vulnerabilities.append(current_vuln)
        
        return vulnerabilities
    
    def _calculate_metrics(self, expected: List[Dict], ai_results: Dict, static_results: Dict) -> Dict[str, Any]:
        """Calculate performance metrics for all tools."""
        metrics = {}
        
        # AI metrics
        ai_vulnerabilities = ai_results.get("vulnerabilities", [])
        ai_metrics = self._calculate_tool_metrics(expected, ai_vulnerabilities)
        metrics["ai"] = ai_metrics
        
        # Static analysis metrics
        for tool_name, tool_result in static_results.items():
            tool_metrics = self._calculate_tool_metrics(expected, tool_result.vulnerabilities_found)
            tool_metrics.update({
                "execution_time": tool_result.execution_time,
                "tool_name": tool_name
            })
            metrics[tool_name] = tool_metrics
        
        return metrics
    
    def _calculate_tool_metrics(self, expected: List[Dict], detected: List[Dict]) -> Dict[str, Any]:
        """Calculate metrics for a single tool."""
        expected_types = {vuln.get("type", "").lower() for vuln in expected}
        detected_types = {vuln.get("type", "").lower() for vuln in detected}
        
        true_positives = len(expected_types.intersection(detected_types))
        false_positives = len(detected_types - expected_types)
        false_negatives = len(expected_types - detected_types)
        true_negatives = 1 if not expected_types and not detected_types else 0
        
        total_predictions = true_positives + false_positives + true_negatives + false_negatives
        
        accuracy = (true_positives + true_negatives) / total_predictions if total_predictions > 0 else 0
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "true_negatives": true_negatives,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }
    
    def _determine_best_tool(self, metrics: Dict[str, Any]) -> str:
        """Determine the best performing tool based on F1 score."""
        best_tool = "ai"
        best_f1 = metrics.get("ai", {}).get("f1_score", 0)
        
        for tool_name, tool_metrics in metrics.items():
            if tool_name != "ai" and isinstance(tool_metrics, dict):
                f1_score = tool_metrics.get("f1_score", 0)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_tool = tool_name
        
        return best_tool
    
    def generate_report(self, results: List[BenchmarkResult] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report."""
        if results is None:
            results = self.results
        
        if not results:
            logger.error("No benchmark results available")
            return {}
        
        logger.info("Generating benchmark report...")
        
        # Generate different report formats
        report_data = {
            "summary": self._generate_summary(results),
            "detailed_results": [asdict(result) for result in results],
            "comparison_charts": self._generate_comparison_charts(results),
            "performance_analysis": self._analyze_performance(results),
            "recommendations": self._generate_recommendations(results),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save reports
        self._save_json_report(report_data)
        self._save_html_report(report_data)
        self._save_pdf_report(report_data)
        
        logger.info("Benchmark report generated successfully")
        return report_data
    
    def _generate_summary(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Generate summary statistics."""
        total_contracts = len(results)
        
        # Tool performance summary
        tool_performance = {}
        for result in results:
            for tool_name, metrics in result.performance_metrics.items():
                if tool_name not in tool_performance:
                    tool_performance[tool_name] = {
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1_score": [],
                        "execution_time": []
                    }
                
                tool_performance[tool_name]["accuracy"].append(metrics.get("accuracy", 0))
                tool_performance[tool_name]["precision"].append(metrics.get("precision", 0))
                tool_performance[tool_name]["recall"].append(metrics.get("recall", 0))
                tool_performance[tool_name]["f1_score"].append(metrics.get("f1_score", 0))
                tool_performance[tool_name]["execution_time"].append(metrics.get("execution_time", 0))
        
        # Calculate averages
        summary = {
            "total_contracts": total_contracts,
            "tool_performance": {},
            "best_overall_tool": "ai",
            "average_accuracy": 0
        }
        
        best_avg_f1 = 0
        total_accuracy = 0
        
        for tool_name, metrics in tool_performance.items():
            avg_metrics = {
                "accuracy": sum(metrics["accuracy"]) / len(metrics["accuracy"]),
                "precision": sum(metrics["precision"]) / len(metrics["precision"]),
                "recall": sum(metrics["recall"]) / len(metrics["recall"]),
                "f1_score": sum(metrics["f1_score"]) / len(metrics["f1_score"]),
                "avg_execution_time": sum(metrics["execution_time"]) / len(metrics["execution_time"])
            }
            
            summary["tool_performance"][tool_name] = avg_metrics
            total_accuracy += avg_metrics["accuracy"]
            
            if avg_metrics["f1_score"] > best_avg_f1:
                best_avg_f1 = avg_metrics["f1_score"]
                summary["best_overall_tool"] = tool_name
        
        summary["average_accuracy"] = total_accuracy / len(tool_performance) if tool_performance else 0
        
        return summary
    
    def _generate_comparison_charts(self, results: List[BenchmarkResult]) -> Dict[str, str]:
        """Generate comparison charts."""
        charts = {}
        
        # Tool comparison chart
        tool_names = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        # Aggregate metrics across all contracts
        tool_metrics = {}
        for result in results:
            for tool_name, metrics in result.performance_metrics.items():
                if tool_name not in tool_metrics:
                    tool_metrics[tool_name] = {
                        "accuracy": [],
                        "precision": [],
                        "recall": [],
                        "f1_score": []
                    }
                
                tool_metrics[tool_name]["accuracy"].append(metrics.get("accuracy", 0))
                tool_metrics[tool_name]["precision"].append(metrics.get("precision", 0))
                tool_metrics[tool_name]["recall"].append(metrics.get("recall", 0))
                tool_metrics[tool_name]["f1_score"].append(metrics.get("f1_score", 0))
        
        # Calculate averages
        for tool_name, metrics in tool_metrics.items():
            tool_names.append(tool_name)
            accuracy_scores.append(sum(metrics["accuracy"]) / len(metrics["accuracy"]))
            precision_scores.append(sum(metrics["precision"]) / len(metrics["precision"]))
            recall_scores.append(sum(metrics["recall"]) / len(metrics["recall"]))
            f1_scores.append(sum(metrics["f1_score"]) / len(metrics["f1_score"]))
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(name='Accuracy', x=tool_names, y=accuracy_scores),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(name='Precision', x=tool_names, y=precision_scores),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(name='Recall', x=tool_names, y=recall_scores),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Bar(name='F1-Score', x=tool_names, y=f1_scores),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Tool Performance Comparison",
            showlegend=False,
            height=800
        )
        
        # Save chart
        chart_file = self.output_dir / f"tool_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(chart_file))
        charts["tool_comparison"] = str(chart_file)
        
        return charts
    
    def _analyze_performance(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Analyze performance characteristics."""
        analysis = {
            "execution_time_analysis": {},
            "accuracy_trends": {},
            "vulnerability_detection_patterns": {}
        }
        
        # Execution time analysis
        execution_times = {}
        for result in results:
            for tool_name, metrics in result.performance_metrics.items():
                if tool_name not in execution_times:
                    execution_times[tool_name] = []
                execution_times[tool_name].append(metrics.get("execution_time", 0))
        
        for tool_name, times in execution_times.items():
            analysis["execution_time_analysis"][tool_name] = {
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "median": sorted(times)[len(times) // 2]
            }
        
        return analysis
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Analyze tool performance
        tool_performance = {}
        for result in results:
            for tool_name, metrics in result.performance_metrics.items():
                if tool_name not in tool_performance:
                    tool_performance[tool_name] = []
                tool_performance[tool_name].append(metrics.get("f1_score", 0))
        
        # Calculate average F1 scores
        avg_f1_scores = {}
        for tool_name, scores in tool_performance.items():
            avg_f1_scores[tool_name] = sum(scores) / len(scores)
        
        # Generate recommendations
        if avg_f1_scores.get("ai", 0) > 0.8:
            recommendations.append("AI model shows excellent performance. Consider deploying for production use.")
        elif avg_f1_scores.get("ai", 0) > 0.6:
            recommendations.append("AI model shows good performance but could benefit from additional training data.")
        else:
            recommendations.append("AI model performance needs improvement. Consider retraining with more diverse data.")
        
        # Compare with static analysis tools
        best_static_tool = max(
            [(tool, score) for tool, score in avg_f1_scores.items() if tool != "ai"],
            key=lambda x: x[1],
            default=("none", 0)
        )
        
        if best_static_tool[1] > avg_f1_scores.get("ai", 0):
            recommendations.append(f"Static analysis tool {best_static_tool[0]} outperforms AI model. Consider hybrid approach.")
        
        # Performance recommendations
        execution_times = {}
        for result in results:
            for tool_name, metrics in result.performance_metrics.items():
                if tool_name not in execution_times:
                    execution_times[tool_name] = []
                execution_times[tool_name].append(metrics.get("execution_time", 0))
        
        for tool_name, times in execution_times.items():
            avg_time = sum(times) / len(times)
            if avg_time > 10:  # More than 10 seconds
                recommendations.append(f"{tool_name} has slow execution time ({avg_time:.1f}s). Consider optimization.")
        
        return recommendations
    
    def _save_json_report(self, report_data: Dict[str, Any]):
        """Save JSON report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        
        with open(json_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"JSON report saved to {json_file}")
    
    def _save_html_report(self, report_data: Dict[str, Any]):
        """Save HTML report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = self.output_dir / f"benchmark_report_{timestamp}.html"
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Contract Vulnerability Detection Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .summary { margin: 20px 0; }
        .tool-comparison { margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        .recommendations { background-color: #e8f4f8; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Smart Contract Vulnerability Detection Benchmark Report</h1>
        <p>Generated on: {{ timestamp }}</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <p>Total Contracts Analyzed: {{ summary.total_contracts }}</p>
        <p>Best Overall Tool: {{ summary.best_overall_tool }}</p>
        <p>Average Accuracy: {{ "%.2f"|format(summary.average_accuracy) }}</p>
    </div>
    
    <div class="tool-comparison">
        <h2>Tool Performance Comparison</h2>
        <table>
            <tr>
                <th>Tool</th>
                <th>Accuracy</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
            </tr>
            {% for tool_name, metrics in summary.tool_performance.items() %}
            <tr>
                <td>{{ tool_name }}</td>
                <td>{{ "%.3f"|format(metrics.accuracy) }}</td>
                <td>{{ "%.3f"|format(metrics.precision) }}</td>
                <td>{{ "%.3f"|format(metrics.recall) }}</td>
                <td>{{ "%.3f"|format(metrics.f1_score) }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {% for recommendation in recommendations %}
            <li>{{ recommendation }}</li>
            {% endfor %}
        </ul>
    </div>
</body>
</html>
        """
        
        template = Template(html_template)
        html_content = template.render(
            timestamp=report_data["timestamp"],
            summary=report_data["summary"],
            recommendations=report_data["recommendations"]
        )
        
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML report saved to {html_file}")
    
    def _save_pdf_report(self, report_data: Dict[str, Any]):
        """Save PDF report."""
        # This would require additional libraries like weasyprint or reportlab
        # For now, we'll create a simple text report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_file = self.output_dir / f"benchmark_report_{timestamp}.txt"
        
        with open(txt_file, 'w') as f:
            f.write("Smart Contract Vulnerability Detection Benchmark Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Generated on: {report_data['timestamp']}\n\n")
            
            f.write("SUMMARY\n")
            f.write("-" * 20 + "\n")
            summary = report_data["summary"]
            f.write(f"Total Contracts: {summary['total_contracts']}\n")
            f.write(f"Best Tool: {summary['best_overall_tool']}\n")
            f.write(f"Average Accuracy: {summary['average_accuracy']:.3f}\n\n")
            
            f.write("TOOL PERFORMANCE\n")
            f.write("-" * 20 + "\n")
            for tool_name, metrics in summary["tool_performance"].items():
                f.write(f"\n{tool_name.upper()}:\n")
                f.write(f"  Accuracy: {metrics['accuracy']:.3f}\n")
                f.write(f"  Precision: {metrics['precision']:.3f}\n")
                f.write(f"  Recall: {metrics['recall']:.3f}\n")
                f.write(f"  F1-Score: {metrics['f1_score']:.3f}\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for i, rec in enumerate(report_data["recommendations"], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Text report saved to {txt_file}")


def run_benchmark_report(api_endpoint: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Convenience function to run benchmark and generate report.
    
    Args:
        api_endpoint: API endpoint for AI analysis
        
    Returns:
        Benchmark report data
    """
    generator = BenchmarkReportGenerator()
    
    # Run benchmark
    results = generator.run_benchmark(api_endpoint)
    
    # Generate report
    report = generator.generate_report(results)
    
    return report


if __name__ == "__main__":
    # Example usage
    report = run_benchmark_report()
    print("Benchmark report generated!")
    print(f"Best tool: {report['summary']['best_overall_tool']}")
    print(f"Average accuracy: {report['summary']['average_accuracy']:.3f}")
