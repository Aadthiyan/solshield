#!/usr/bin/env python3
"""
Load Testing with Locust

This module provides comprehensive load testing for the smart contract
vulnerability detection API using Locust.
"""

from locust import HttpUser, task, between, events
import random
import json
import time
from typing import Dict, Any

class SmartContractUser(HttpUser):
    """Locust user class for smart contract vulnerability detection API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        self.contracts = self._load_test_contracts()
        self.report_ids = []
    
    def _load_test_contracts(self) -> list:
        """Load test contracts with different characteristics"""
        return [
            # Safe contract
            {
                "contract_code": """
                contract SafeContract {
                    uint256 public balance;
                    address public owner;
                    
                    modifier onlyOwner() {
                        require(msg.sender == owner, "Not the owner");
                        _;
                    }
                    
                    function deposit() public payable {
                        balance += msg.value;
                    }
                    
                    function withdraw(uint256 amount) public onlyOwner {
                        require(amount <= balance, "Insufficient balance");
                        balance -= amount;
                        payable(owner).transfer(amount);
                    }
                }
                """,
                "contract_name": "SafeContract",
                "expected_vulnerabilities": 0
            },
            
            # Vulnerable contract - Reentrancy
            {
                "contract_code": """
                contract VulnerableContract {
                    mapping(address => uint256) public balances;
                    
                    function withdraw() public {
                        uint256 amount = balances[msg.sender];
                        require(amount > 0, "No balance");
                        balances[msg.sender] = 0;
                        msg.sender.transfer(amount);
                    }
                    
                    function deposit() public payable {
                        balances[msg.sender] += msg.value;
                    }
                }
                """,
                "contract_name": "VulnerableContract",
                "expected_vulnerabilities": 1
            },
            
            # Vulnerable contract - Integer Overflow
            {
                "contract_code": """
                contract IntegerOverflowContract {
                    uint256 public totalSupply;
                    
                    function add(uint256 a, uint256 b) public pure returns (uint256) {
                        return a + b;
                    }
                    
                    function multiply(uint256 a, uint256 b) public pure returns (uint256) {
                        return a * b;
                    }
                }
                """,
                "contract_name": "IntegerOverflowContract",
                "expected_vulnerabilities": 1
            },
            
            # Vulnerable contract - Access Control
            {
                "contract_code": """
                contract AccessControlVulnerable {
                    uint256 public secret;
                    
                    function setSecret(uint256 _secret) public {
                        secret = _secret;
                    }
                    
                    function getSecret() public view returns (uint256) {
                        return secret;
                    }
                }
                """,
                "contract_name": "AccessControlVulnerable",
                "expected_vulnerabilities": 1
            },
            
            # Complex contract
            {
                "contract_code": """
                contract ComplexContract {
                    struct User {
                        uint256 balance;
                        bool isActive;
                        uint256 lastActivity;
                    }
                    
                    mapping(address => User) public users;
                    address public owner;
                    
                    modifier onlyOwner() {
                        require(msg.sender == owner, "Not the owner");
                        _;
                    }
                    
                    function createUser() public {
                        users[msg.sender] = User({
                            balance: 0,
                            isActive: true,
                            lastActivity: block.timestamp
                        });
                    }
                    
                    function deposit() public payable {
                        require(users[msg.sender].isActive, "User not active");
                        users[msg.sender].balance += msg.value;
                        users[msg.sender].lastActivity = block.timestamp;
                    }
                    
                    function withdraw(uint256 amount) public {
                        require(users[msg.sender].isActive, "User not active");
                        require(users[msg.sender].balance >= amount, "Insufficient balance");
                        users[msg.sender].balance -= amount;
                        users[msg.sender].lastActivity = block.timestamp;
                        payable(msg.sender).transfer(amount);
                    }
                }
                """,
                "contract_name": "ComplexContract",
                "expected_vulnerabilities": 0
            }
        ]
    
    @task(3)
    def analyze_contract(self):
        """Test contract analysis endpoint"""
        contract = random.choice(self.contracts)
        
        request_data = {
            "contract_code": contract["contract_code"],
            "contract_name": contract["contract_name"],
            "model_type": random.choice(["codebert", "gnn", "ensemble"]),
            "include_optimization_suggestions": random.choice([True, False]),
            "include_explanation": random.choice([True, False])
        }
        
        with self.client.post(
            "/api/v1/analyze",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.report_ids.append(data.get("report_id"))
                    response.success()
                else:
                    response.failure(f"Analysis failed: {data.get('message')}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def get_report(self):
        """Test report retrieval endpoint"""
        if not self.report_ids:
            return
        
        report_id = random.choice(self.report_ids)
        
        with self.client.get(
            f"/api/v1/report/{report_id}",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success") and "report" in data:
                    response.success()
                else:
                    response.failure(f"Report retrieval failed: {data.get('message')}")
            elif response.status_code == 404:
                # Report not found, remove from list
                if report_id in self.report_ids:
                    self.report_ids.remove(report_id)
                response.success()  # Don't count as failure
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def batch_analysis(self):
        """Test batch analysis endpoint"""
        # Select 2-3 random contracts
        num_contracts = random.randint(2, 3)
        selected_contracts = random.sample(self.contracts, num_contracts)
        
        contracts_data = []
        for contract in selected_contracts:
            contracts_data.append({
                "contract_code": contract["contract_code"],
                "contract_name": contract["contract_name"],
                "model_type": random.choice(["codebert", "gnn", "ensemble"])
            })
        
        request_data = {
            "contracts": contracts_data,
            "batch_id": f"batch_{int(time.time())}"
        }
        
        with self.client.post(
            "/api/v1/analyze/batch",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    # Store batch contract IDs
                    self.report_ids.extend(data.get("contract_ids", []))
                    response.success()
                else:
                    response.failure(f"Batch analysis failed: {data.get('message')}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def health_check(self):
        """Test health check endpoint"""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") in ["healthy", "degraded"]:
                    response.success()
                else:
                    response.failure(f"Unhealthy status: {data.get('status')}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def system_status(self):
        """Test system status endpoint"""
        with self.client.get("/api/v1/status", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "status" in data and "models" in data:
                    response.success()
                else:
                    response.failure("Invalid status response format")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def get_metrics(self):
        """Test metrics endpoint"""
        with self.client.get("/api/v1/metrics", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "total_requests" in data and "successful_requests" in data:
                    response.success()
                else:
                    response.failure("Invalid metrics response format")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")
    
    @task(1)
    def list_reports(self):
        """Test list reports endpoint"""
        with self.client.get("/api/v1/reports", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "success" in data and "reports" in data:
                    response.success()
                else:
                    response.failure("Invalid reports response format")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

class HighLoadUser(HttpUser):
    """High load user for stress testing"""
    
    wait_time = between(0.1, 0.5)  # Very short wait time
    
    def on_start(self):
        """Called when a user starts"""
        self.contracts = [
            "contract Test { function test() public {} }",
            "contract Test2 { function test() public {} }",
            "contract Test3 { function test() public {} }"
        ]
    
    @task(5)
    def rapid_analysis(self):
        """Rapid contract analysis"""
        contract_code = random.choice(self.contracts)
        
        request_data = {
            "contract_code": contract_code,
            "model_type": "ensemble"
        }
        
        with self.client.post(
            "/api/v1/analyze",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check"""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

class BatchLoadUser(HttpUser):
    """User focused on batch processing"""
    
    wait_time = between(2, 5)  # Longer wait time for batch processing
    
    def on_start(self):
        """Called when a user starts"""
        self.contracts = [
            "contract Batch1 { function test() public {} }",
            "contract Batch2 { function test() public {} }",
            "contract Batch3 { function test() public {} }",
            "contract Batch4 { function test() public {} }",
            "contract Batch5 { function test() public {} }"
        ]
    
    @task(3)
    def batch_analysis(self):
        """Batch analysis with maximum contracts"""
        # Use maximum number of contracts (10)
        contracts_data = []
        for i in range(10):
            contracts_data.append({
                "contract_code": f"contract Batch{i} {{ function test() public {{}} }}",
                "model_type": random.choice(["codebert", "gnn", "ensemble"])
            })
        
        request_data = {
            "contracts": contracts_data,
            "batch_id": f"batch_{int(time.time())}_{random.randint(1000, 9999)}"
        }
        
        with self.client.post(
            "/api/v1/analyze/batch",
            json=request_data,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check"""
        with self.client.get("/api/v1/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

# Custom event handlers
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Custom request event handler"""
    if exception:
        print(f"Request failed: {name} - {exception}")
    else:
        print(f"Request completed: {name} - {response_time}ms")

@events.user_error.add_listener
def on_user_error(user_instance, exception, tb, **kwargs):
    """Custom user error handler"""
    print(f"User error: {exception}")

# Load test scenarios
class LoadTestScenarios:
    """Predefined load test scenarios"""
    
    @staticmethod
    def light_load():
        """Light load scenario"""
        return {
            "users": 10,
            "spawn_rate": 2,
            "host": "http://localhost:8000",
            "duration": "2m"
        }
    
    @staticmethod
    def medium_load():
        """Medium load scenario"""
        return {
            "users": 50,
            "spawn_rate": 5,
            "host": "http://localhost:8000",
            "duration": "5m"
        }
    
    @staticmethod
    def heavy_load():
        """Heavy load scenario"""
        return {
            "users": 100,
            "spawn_rate": 10,
            "host": "http://localhost:8000",
            "duration": "10m"
        }
    
    @staticmethod
    def stress_test():
        """Stress test scenario"""
        return {
            "users": 200,
            "spawn_rate": 20,
            "host": "http://localhost:8000",
            "duration": "15m"
        }

# Performance thresholds
PERFORMANCE_THRESHOLDS = {
    "max_response_time": 30.0,  # 30 seconds
    "min_success_rate": 0.95,   # 95% success rate
    "max_error_rate": 0.05      # 5% error rate
}

# Run load test
if __name__ == "__main__":
    import subprocess
    import sys
    
    # Default to light load if no arguments
    scenario = sys.argv[1] if len(sys.argv) > 1 else "light"
    
    scenarios = {
        "light": LoadTestScenarios.light_load(),
        "medium": LoadTestScenarios.medium_load(),
        "heavy": LoadTestScenarios.heavy_load(),
        "stress": LoadTestScenarios.stress_test()
    }
    
    if scenario not in scenarios:
        print(f"Unknown scenario: {scenario}")
        print(f"Available scenarios: {list(scenarios.keys())}")
        sys.exit(1)
    
    config = scenarios[scenario]
    
    # Build locust command
    cmd = [
        "locust",
        "-f", __file__,
        "--host", config["host"],
        "--users", str(config["users"]),
        "--spawn-rate", str(config["spawn_rate"]),
        "--run-time", config["duration"],
        "--headless"
    ]
    
    print(f"Running {scenario} load test: {config}")
    subprocess.run(cmd)
