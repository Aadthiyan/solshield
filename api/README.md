# Smart Contract Vulnerability Detection API

A comprehensive FastAPI-based backend for detecting vulnerabilities in smart contracts using state-of-the-art machine learning models including CodeBERT and Graph Neural Networks.

## 🚀 Features

- **Vulnerability Detection**: Analyze Solidity contracts for security vulnerabilities
- **Multiple Models**: Support for CodeBERT, GNN, and ensemble predictions
- **Detailed Reports**: Comprehensive vulnerability reports with explanations and recommendations
- **Gas Optimization**: Suggestions for gas-efficient contract implementations
- **Batch Processing**: Analyze multiple contracts simultaneously
- **Real-time Monitoring**: Health checks, metrics, and system status
- **Load Testing**: Comprehensive performance testing with Locust
- **Comprehensive Logging**: Structured logging with performance monitoring

## 📋 API Endpoints

### Vulnerability Detection

#### Analyze Contract
```http
POST /api/v1/analyze
```

Analyze a smart contract for vulnerabilities.

**Request Body:**
```json
{
  "contract_code": "contract Test { function test() public {} }",
  "contract_name": "TestContract",
  "model_type": "ensemble",
  "include_optimization_suggestions": true,
  "include_explanation": true
}
```

**Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "report_id": "uuid-string",
  "estimated_processing_time": 2.5
}
```

#### Get Report
```http
GET /api/v1/report/{report_id}
```

Retrieve a vulnerability report by ID.

**Response:**
```json
{
  "success": true,
  "message": "Report retrieved successfully",
  "report": {
    "report_id": "uuid-string",
    "timestamp": "2024-01-01T00:00:00Z",
    "contract_name": "TestContract",
    "is_vulnerable": true,
    "overall_confidence": 0.85,
    "risk_score": 7.5,
    "vulnerabilities": [
      {
        "type": "reentrancy",
        "severity": "high",
        "confidence": 0.9,
        "description": "Reentrancy vulnerability detected",
        "location": "Line 5: msg.sender.transfer(amount);",
        "explanation": "External call before state change",
        "recommendation": "Use checks-effects-interactions pattern"
      }
    ],
    "optimization_suggestions": [
      {
        "type": "storage_optimization",
        "description": "Consider using smaller data types",
        "potential_savings": "10-30%",
        "priority": "medium"
      }
    ]
  }
}
```

#### Batch Analysis
```http
POST /api/v1/analyze/batch
```

Analyze multiple contracts in batch.

**Request Body:**
```json
{
  "contracts": [
    {
      "contract_code": "contract Test1 { function test() public {} }",
      "model_type": "ensemble"
    },
    {
      "contract_code": "contract Test2 { function test() public {} }",
      "model_type": "codebert"
    }
  ],
  "batch_id": "batch-123"
}
```

#### Batch Reports
```http
GET /api/v1/batch/{batch_id}
```

Retrieve all reports for a batch analysis.

### System Management

#### Health Check
```http
GET /api/v1/health
```

Check API health status.

#### System Status
```http
GET /api/v1/status
```

Get detailed system status including model information and performance metrics.

#### Metrics
```http
GET /api/v1/metrics
```

Get performance metrics and statistics.

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install API Dependencies

```bash
pip install fastapi uvicorn pytest httpx locust
```

## 🚀 Quick Start

### Start the API Server

```bash
# Development server
python api/main.py

# Production server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Test the API

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Analyze a contract
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "contract_code": "contract Test { function test() public {} }",
    "model_type": "ensemble"
  }'
```

## 📊 Model Types

### CodeBERT
- **Type**: Transformer-based model
- **Use Case**: Code understanding and vulnerability detection
- **Strengths**: Excellent at understanding code semantics
- **Best For**: Complex contracts with multiple functions

### GNN (Graph Neural Network)
- **Type**: Graph-based model
- **Use Case**: Structural analysis of contract relationships
- **Strengths**: Captures code structure and dependencies
- **Best For**: Contracts with complex control flow

### Ensemble
- **Type**: Combined predictions
- **Use Case**: Maximum accuracy and reliability
- **Strengths**: Combines strengths of both models
- **Best For**: Production use and critical applications

## 🔍 Vulnerability Types

The API detects the following vulnerability types:

- **Reentrancy**: External calls that can be re-entered
- **Integer Overflow**: Arithmetic operations that can overflow
- **Access Control**: Improper authorization mechanisms
- **Unchecked External Calls**: Missing error handling for external calls
- **Front-running**: Transaction ordering vulnerabilities
- **Timestamp Dependence**: Block timestamp manipulation
- **Gas Limit**: Gas consumption and limit issues
- **Denial of Service**: Contract failure and gas griefing
- **Transaction Origin**: Authorization bypass vulnerabilities

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest api/tests/test_api.py -v

# Run specific test
pytest api/tests/test_api.py::TestContractAnalysis::test_analyze_contract_success -v
```

### Load Testing

#### Using Locust

```bash
# Install Locust
pip install locust

# Run load test
locust -f api/load_tests/locustfile.py --host http://localhost:8000

# Run specific scenario
python api/load_tests/locustfile.py light
```

#### Using Performance Test Script

```bash
# Run concurrent test
python api/load_tests/performance_test.py --test concurrent --requests 100 --concurrency 10

# Run stress test
python api/load_tests/performance_test.py --test stress --duration 60

# Run all tests
python api/load_tests/performance_test.py --test all
```

## 📈 Performance Monitoring

### Metrics Available

- **Request Count**: Total, successful, and failed requests
- **Response Time**: Min, max, mean, and median response times
- **Throughput**: Requests per second
- **Model Usage**: Usage statistics for each model
- **Vulnerability Statistics**: Detection rates by vulnerability type

### Monitoring Endpoints

```bash
# Get current metrics
curl http://localhost:8000/api/v1/metrics

# Get system status
curl http://localhost:8000/api/v1/status

# Get model information
curl http://localhost:8000/api/v1/models
```

## 🔧 Configuration

### Environment Variables

```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export API_WORKERS=4

# Model Configuration
export MODEL_DIR=models
export CACHE_SIZE=1000
export MAX_BATCH_SIZE=10

# Logging Configuration
export LOG_LEVEL=INFO
export LOG_FILE=logs/api.log
```

### Model Loading

The API automatically loads available models from the `models/` directory:

- `models/codebert_output/` - CodeBERT model
- `models/gnn_output/` - GNN model

## 📝 Logging

### Log Files

- `logs/api.log` - General API logs
- `logs/errors.log` - Error logs
- `logs/performance.log` - Performance metrics

### Log Levels

- **DEBUG**: Detailed debugging information
- **INFO**: General information about API operations
- **WARNING**: Warning messages
- **ERROR**: Error messages
- **CRITICAL**: Critical errors

### Structured Logging

The API uses structured logging with the following fields:

- `timestamp`: ISO format timestamp
- `level`: Log level
- `message`: Log message
- `request_id`: Unique request identifier
- `method`: HTTP method
- `url`: Request URL
- `status_code`: Response status code
- `response_time`: Response time in seconds

## 🚀 Deployment

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Production Considerations

1. **Load Balancing**: Use multiple API instances behind a load balancer
2. **Caching**: Implement Redis caching for frequently accessed reports
3. **Database**: Use PostgreSQL for persistent storage
4. **Monitoring**: Implement Prometheus metrics and Grafana dashboards
5. **Security**: Use HTTPS, API keys, and rate limiting

## 🔒 Security

### API Security

- **Input Validation**: All inputs are validated using Pydantic models
- **Rate Limiting**: Implement rate limiting to prevent abuse
- **Authentication**: Add API key authentication for production
- **HTTPS**: Use HTTPS in production environments

### Model Security

- **Input Sanitization**: Contract code is sanitized before processing
- **Model Isolation**: Models run in isolated environments
- **Resource Limits**: Memory and CPU limits for model inference

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

### Example Usage

```python
import requests

# Analyze a contract
response = requests.post(
    "http://localhost:8000/api/v1/analyze",
    json={
        "contract_code": """
        contract VulnerableContract {
            function withdraw() public {
                msg.sender.transfer(address(this).balance);
            }
        }
        """,
        "model_type": "ensemble"
    }
)

result = response.json()
print(f"Report ID: {result['report_id']}")

# Get the report
report_response = requests.get(
    f"http://localhost:8000/api/v1/report/{result['report_id']}"
)

report = report_response.json()
print(f"Vulnerabilities found: {len(report['report']['vulnerabilities'])}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

1. Check the API documentation at `/docs`
2. Review the logs in `logs/` directory
3. Run the health check endpoint
4. Check system status and metrics

## 🔄 Changelog

### Version 1.0.0
- Initial release
- CodeBERT and GNN model support
- Batch processing
- Comprehensive testing
- Performance monitoring
- Load testing with Locust
