#!/usr/bin/env python3
"""
API Unit Tests using pytest

This module contains comprehensive unit tests for the FastAPI application
including endpoint testing, request/response validation, and error handling.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any
from fastapi.testclient import TestClient
from fastapi import FastAPI
import tempfile
import shutil
from pathlib import Path

# Add project directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.main import app
from api.models.schemas import (
    ContractSubmissionRequest, ModelType, VulnerabilityType,
    VulnerabilitySeverity
)

# Test client
client = TestClient(app)

class TestAPIBasic:
    """Basic API functionality tests"""
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_api_health_check(self):
        """Test API health check endpoint"""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "models_loaded" in data

class TestContractAnalysis:
    """Contract analysis endpoint tests"""
    
    def test_analyze_contract_success(self):
        """Test successful contract analysis"""
        contract_code = """
        contract TestContract {
            uint256 public balance;
            
            function deposit() public payable {
                balance += msg.value;
            }
            
            function withdraw(uint256 amount) public {
                require(amount <= balance, "Insufficient balance");
                balance -= amount;
                payable(msg.sender).transfer(amount);
            }
        }
        """
        
        request_data = {
            "contract_code": contract_code,
            "contract_name": "TestContract",
            "model_type": "ensemble",
            "include_optimization_suggestions": True,
            "include_explanation": True
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "report_id" in data
        assert "message" in data
    
    def test_analyze_contract_empty_code(self):
        """Test contract analysis with empty code"""
        request_data = {
            "contract_code": "",
            "model_type": "ensemble"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 400
    
    def test_analyze_contract_invalid_code(self):
        """Test contract analysis with invalid code"""
        request_data = {
            "contract_code": "This is not Solidity code",
            "model_type": "ensemble"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 400
    
    def test_analyze_contract_different_models(self):
        """Test contract analysis with different model types"""
        contract_code = """
        contract VulnerableContract {
            function withdraw() public {
                msg.sender.transfer(address(this).balance);
            }
        }
        """
        
        for model_type in ["codebert", "gnn", "ensemble"]:
            request_data = {
                "contract_code": contract_code,
                "model_type": model_type
            }
            
            response = client.post("/api/v1/analyze", json=request_data)
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
    
    def test_analyze_contract_missing_fields(self):
        """Test contract analysis with missing required fields"""
        request_data = {
            "model_type": "ensemble"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 422  # Validation error

class TestReportRetrieval:
    """Report retrieval endpoint tests"""
    
    def test_get_report_success(self):
        """Test successful report retrieval"""
        # First, create a report
        contract_code = """
        contract TestContract {
            function test() public {
                require(msg.sender != address(0));
            }
        }
        """
        
        # Submit contract for analysis
        request_data = {
            "contract_code": contract_code,
            "model_type": "ensemble"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        report_id = data["report_id"]
        
        # Retrieve the report
        response = client.get(f"/api/v1/report/{report_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "report" in data
        assert data["report"]["report_id"] == report_id
    
    def test_get_report_not_found(self):
        """Test report retrieval with non-existent ID"""
        response = client.get("/api/v1/report/non-existent-id")
        assert response.status_code == 404
    
    def test_get_report_invalid_id(self):
        """Test report retrieval with invalid ID format"""
        response = client.get("/api/v1/report/invalid-id-format")
        assert response.status_code == 404

class TestBatchProcessing:
    """Batch processing endpoint tests"""
    
    def test_batch_analysis_success(self):
        """Test successful batch analysis"""
        contracts = [
            {
                "contract_code": "contract Test1 { function test() public {} }",
                "model_type": "ensemble"
            },
            {
                "contract_code": "contract Test2 { function test() public {} }",
                "model_type": "codebert"
            }
        ]
        
        request_data = {
            "contracts": contracts,
            "batch_id": "test-batch-1"
        }
        
        response = client.post("/api/v1/analyze/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "batch_id" in data
        assert "contract_ids" in data
        assert len(data["contract_ids"]) == 2
    
    def test_batch_analysis_too_many_contracts(self):
        """Test batch analysis with too many contracts"""
        contracts = []
        for i in range(11):  # More than 10 contracts
            contracts.append({
                "contract_code": f"contract Test{i} {{ function test() public {{}} }}",
                "model_type": "ensemble"
            })
        
        request_data = {
            "contracts": contracts
        }
        
        response = client.post("/api/v1/analyze/batch", json=request_data)
        assert response.status_code == 400
    
    def test_batch_analysis_empty_contracts(self):
        """Test batch analysis with empty contracts list"""
        request_data = {
            "contracts": []
        }
        
        response = client.post("/api/v1/analyze/batch", json=request_data)
        assert response.status_code == 422  # Validation error
    
    def test_get_batch_reports(self):
        """Test batch report retrieval"""
        # First, create a batch
        contracts = [
            {
                "contract_code": "contract Test1 { function test() public {} }",
                "model_type": "ensemble"
            }
        ]
        
        request_data = {
            "contracts": contracts
        }
        
        response = client.post("/api/v1/analyze/batch", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        batch_id = data["batch_id"]
        
        # Retrieve batch reports
        response = client.get(f"/api/v1/batch/{batch_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "reports" in data
        assert "batch_id" in data

class TestSystemEndpoints:
    """System management endpoint tests"""
    
    def test_system_status(self):
        """Test system status endpoint"""
        response = client.get("/api/v1/status")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models" in data
        assert "system_metrics" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/api/v1/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert "total_requests" in data
        assert "successful_requests" in data
        assert "failed_requests" in data
        assert "average_processing_time" in data
    
    def test_reset_metrics(self):
        """Test metrics reset endpoint"""
        response = client.post("/api/v1/metrics/reset")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
    
    def test_models_info(self):
        """Test models information endpoint"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "models" in data
    
    def test_logs_endpoint(self):
        """Test logs endpoint"""
        response = client.get("/api/v1/logs")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "logs" in data
    
    def test_version_endpoint(self):
        """Test version endpoint"""
        response = client.get("/api/v1/version")
        assert response.status_code == 200
        
        data = response.json()
        assert "success" in data
        assert "version" in data
        assert "api_version" in data["version"]

class TestErrorHandling:
    """Error handling tests"""
    
    def test_invalid_json(self):
        """Test invalid JSON in request body"""
        response = client.post(
            "/api/v1/analyze",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
    
    def test_missing_content_type(self):
        """Test missing content type header"""
        response = client.post("/api/v1/analyze", data="{}")
        assert response.status_code == 422
    
    def test_invalid_model_type(self):
        """Test invalid model type"""
        request_data = {
            "contract_code": "contract Test { function test() public {} }",
            "model_type": "invalid_model"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 422
    
    def test_large_contract_code(self):
        """Test very large contract code"""
        large_code = "contract Test { " + "function test() public {} " * 1000 + "}"
        
        request_data = {
            "contract_code": large_code,
            "model_type": "ensemble"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        # Should either succeed or fail gracefully
        assert response.status_code in [200, 413, 422]

class TestConcurrentRequests:
    """Concurrent request handling tests"""
    
    def test_concurrent_analysis_requests(self):
        """Test handling of concurrent analysis requests"""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request():
            contract_code = f"""
            contract TestContract {{
                function test() public {{
                    require(msg.sender != address(0));
                }}
            }}
            """
            
            request_data = {
                "contract_code": contract_code,
                "model_type": "ensemble"
            }
            
            response = client.post("/api/v1/analyze", json=request_data)
            results.put(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
        
        # All requests should succeed
        assert all(status == 200 for status in status_codes)
        assert len(status_codes) == 5

class TestDataValidation:
    """Data validation tests"""
    
    def test_contract_name_validation(self):
        """Test contract name validation"""
        request_data = {
            "contract_code": "contract Test { function test() public {} }",
            "contract_name": "",  # Empty name should be handled
            "model_type": "ensemble"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 422
    
    def test_model_type_validation(self):
        """Test model type validation"""
        request_data = {
            "contract_code": "contract Test { function test() public {} }",
            "model_type": "invalid_type"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 422
    
    def test_boolean_field_validation(self):
        """Test boolean field validation"""
        request_data = {
            "contract_code": "contract Test { function test() public {} }",
            "model_type": "ensemble",
            "include_optimization_suggestions": "not_a_boolean"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        assert response.status_code == 422

class TestPerformance:
    """Performance tests"""
    
    def test_response_time(self):
        """Test API response time"""
        start_time = time.time()
        
        request_data = {
            "contract_code": "contract Test { function test() public {} }",
            "model_type": "ensemble"
        }
        
        response = client.post("/api/v1/analyze", json=request_data)
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # Response should be reasonably fast (adjust threshold as needed)
        assert response_time < 30.0  # 30 seconds max
    
    def test_memory_usage(self):
        """Test memory usage doesn't grow excessively"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Make multiple requests
        for _ in range(10):
            request_data = {
                "contract_code": "contract Test { function test() public {} }",
                "model_type": "ensemble"
            }
            
            response = client.post("/api/v1/analyze", json=request_data)
            assert response.status_code == 200
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust threshold as needed)
        assert memory_increase < 100 * 1024 * 1024  # 100MB max increase

# Pytest fixtures
@pytest.fixture(scope="session")
def test_app():
    """Test application fixture"""
    return app

@pytest.fixture(scope="session")
def test_client():
    """Test client fixture"""
    return TestClient(app)

@pytest.fixture
def sample_contract():
    """Sample contract fixture"""
    return """
    contract SampleContract {
        uint256 public balance;
        
        function deposit() public payable {
            balance += msg.value;
        }
        
        function withdraw(uint256 amount) public {
            require(amount <= balance, "Insufficient balance");
            balance -= amount;
            payable(msg.sender).transfer(amount);
        }
    }
    """

@pytest.fixture
def vulnerable_contract():
    """Vulnerable contract fixture"""
    return """
    contract VulnerableContract {
        function withdraw() public {
            msg.sender.transfer(address(this).balance);
        }
    }
    """

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
