# API Documentation

## 🚀 Smart Contract Vulnerability Detection API

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, the API does not require authentication for development purposes. In production, JWT tokens will be implemented.

## 📋 API Endpoints

### Health & Status Endpoints

#### GET /health
Simple health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "message": "API is running"
}
```

#### GET /api/v1/status
Get detailed system status.

**Response:**
```json
{
  "status": "operational",
  "version": "1.0.0",
  "uptime": 3600,
  "models_loaded": {
    "codebert": true,
    "gnn": true,
    "ensemble": true
  },
  "system_resources": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.1
  }
}
```

#### GET /api/v1/metrics
Get performance metrics.

**Response:**
```json
{
  "total_requests": 1250,
  "successful_requests": 1200,
  "failed_requests": 50,
  "average_response_time": 2.5,
  "requests_per_minute": 45,
  "model_performance": {
    "codebert": {
      "average_inference_time": 0.8,
      "accuracy": 0.942
    },
    "gnn": {
      "average_inference_time": 0.6,
      "accuracy": 0.918
    },
    "ensemble": {
      "average_inference_time": 1.2,
      "accuracy": 0.961
    }
  }
}
```

### Contract Analysis Endpoints

#### POST /api/v1/analyze
Analyze a single smart contract for vulnerabilities.

**Request Body:**
```json
{
  "contract_code": "pragma solidity ^0.8.0;\ncontract Test {\n    // contract code here\n}",
  "contract_name": "TestContract",
  "model_type": "ensemble",
  "include_optimization_suggestions": true,
  "include_explanation": true
}
```

**Parameters:**
- `contract_code` (string, required): The Solidity contract source code
- `contract_name` (string, optional): Name of the contract
- `model_type` (string, optional): "codebert", "gnn", or "ensemble" (default: "ensemble")
- `include_optimization_suggestions` (boolean, optional): Include gas optimization suggestions
- `include_explanation` (boolean, optional): Include detailed explanations

**Response:**
```json
{
  "success": true,
  "message": "Analysis completed successfully",
  "report_id": "def065b5-9abe-4420-b374-e51f990ec88b",
  "estimated_processing_time": 0.0003478527069091797
}
```

#### GET /api/v1/report/{report_id}
Get the analysis report for a specific report ID.

**Response:**
```json
{
  "success": true,
  "report": {
    "report_id": "def065b5-9abe-4420-b374-e51f990ec88b",
    "contract_name": "TestContract",
    "analysis_timestamp": "2024-01-15T10:30:00Z",
    "risk_score": 7.5,
    "is_vulnerable": true,
    "vulnerabilities": [
      {
        "id": "vuln_001",
        "type": "reentrancy",
        "severity": "high",
        "description": "Potential reentrancy vulnerability in withdraw function",
        "line_number": 15,
        "confidence": 0.95,
        "recommendation": "Use checks-effects-interactions pattern"
      }
    ],
    "optimization_suggestions": [
      {
        "id": "opt_001",
        "type": "gas_optimization",
        "description": "Consider using uint256 instead of uint8 for better gas efficiency",
        "potential_savings": "15%",
        "line_number": 8
      }
    ],
    "summary": {
      "total_vulnerabilities": 1,
      "critical_vulnerabilities": 0,
      "high_vulnerabilities": 1,
      "medium_vulnerabilities": 0,
      "low_vulnerabilities": 0
    }
  }
}
```

### Batch Analysis Endpoints

#### POST /api/v1/analyze/batch
Analyze multiple contracts in batch.

**Request Body:**
```json
{
  "contracts": [
    {
      "contract_code": "pragma solidity ^0.8.0;\ncontract Contract1 {\n    // code\n}",
      "contract_name": "Contract1"
    },
    {
      "contract_code": "pragma solidity ^0.8.0;\ncontract Contract2 {\n    // code\n}",
      "contract_name": "Contract2"
    }
  ],
  "model_type": "ensemble",
  "include_optimization_suggestions": true
}
```

**Response:**
```json
{
  "success": true,
  "batch_id": "batch_12345",
  "message": "Batch analysis started",
  "total_contracts": 2,
  "estimated_completion_time": 300
}
```

#### GET /api/v1/batch/{batch_id}
Get batch analysis results.

**Response:**
```json
{
  "success": true,
  "batch_id": "batch_12345",
  "status": "completed",
  "total_contracts": 2,
  "completed_contracts": 2,
  "failed_contracts": 0,
  "results": [
    {
      "contract_name": "Contract1",
      "report_id": "report_001",
      "status": "completed",
      "risk_score": 5.2
    },
    {
      "contract_name": "Contract2",
      "report_id": "report_002",
      "status": "completed",
      "risk_score": 8.1
    }
  ]
}
```

### Report Management Endpoints

#### GET /api/v1/reports
List all analysis reports with pagination.

**Query Parameters:**
- `limit` (integer, optional): Number of reports to return (default: 10)
- `offset` (integer, optional): Number of reports to skip (default: 0)
- `sort_by` (string, optional): Sort field ("timestamp", "risk_score", "contract_name")
- `sort_order` (string, optional): Sort order ("asc", "desc")
- `filter_by` (string, optional): Filter by vulnerability type

**Response:**
```json
{
  "success": true,
  "reports": [
    {
      "report_id": "def065b5-9abe-4420-b374-e51f990ec88b",
      "contract_name": "TestContract",
      "analysis_timestamp": "2024-01-15T10:30:00Z",
      "risk_score": 7.5,
      "is_vulnerable": true,
      "vulnerability_count": 3
    }
  ],
  "pagination": {
    "total": 150,
    "limit": 10,
    "offset": 0,
    "has_next": true,
    "has_previous": false
  }
}
```

#### DELETE /api/v1/report/{report_id}
Delete a specific analysis report.

**Response:**
```json
{
  "success": true,
  "message": "Report deleted successfully"
}
```

#### GET /api/v1/reports/stats
Get statistics about all reports.

**Response:**
```json
{
  "success": true,
  "statistics": {
    "total_reports": 150,
    "vulnerable_contracts": 45,
    "average_risk_score": 4.2,
    "vulnerability_distribution": {
      "critical": 5,
      "high": 25,
      "medium": 35,
      "low": 15
    },
    "top_vulnerability_types": [
      {
        "type": "reentrancy",
        "count": 20
      },
      {
        "type": "integer_overflow",
        "count": 15
      }
    ]
  }
}
```

## 🔧 Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "additional error details"
  },
  "request_id": "req_12345"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Request validation failed | 400 |
| `FILE_TOO_LARGE` | Contract file exceeds size limit | 413 |
| `INVALID_CONTRACT` | Contract code is invalid | 422 |
| `MODEL_ERROR` | ML model inference failed | 500 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INTERNAL_ERROR` | Internal server error | 500 |

### Example Error Responses

#### Validation Error
```json
{
  "success": false,
  "error": "Contract code is required",
  "code": "VALIDATION_ERROR",
  "details": {
    "field": "contract_code",
    "message": "This field is required"
  }
}
```

#### Model Error
```json
{
  "success": false,
  "error": "Model inference failed",
  "code": "MODEL_ERROR",
  "details": {
    "model": "codebert",
    "error": "CUDA out of memory"
  }
}
```

## 📊 Rate Limiting

### Rate Limits
- **Analysis Endpoints**: 10 requests per minute per IP
- **Report Endpoints**: 100 requests per minute per IP
- **Health Endpoints**: 1000 requests per minute per IP

### Rate Limit Headers
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 7
X-RateLimit-Reset: 1642248000
```

## 🔒 Security Considerations

### Input Validation
- Contract code size limit: 10MB
- File type validation for uploads
- SQL injection prevention
- XSS protection

### CORS Configuration
```json
{
  "allow_origins": ["*"],
  "allow_credentials": true,
  "allow_methods": ["GET", "POST", "DELETE"],
  "allow_headers": ["*"]
}
```

## 📈 Performance Considerations

### Response Times
- **Health Check**: < 100ms
- **Single Analysis**: 2-5 seconds
- **Batch Analysis**: 5-10 seconds per contract
- **Report Retrieval**: < 500ms

### Optimization Tips
1. Use batch analysis for multiple contracts
2. Cache frequently accessed reports
3. Use appropriate model types for your needs
4. Monitor rate limits to avoid throttling

## 🧪 Testing

### Test Endpoints
```bash
# Health check
curl -X GET http://localhost:8000/health

# Analyze contract
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "contract_code": "pragma solidity ^0.8.0;\ncontract Test {\n    function test() public {}\n}",
    "contract_name": "Test"
  }'

# Get report
curl -X GET http://localhost:8000/api/v1/report/{report_id}
```

### Load Testing
```bash
# Install Locust
pip install locust

# Run load tests
locust -f api/load_tests/locustfile.py --host=http://localhost:8000
```

## 📚 SDK Examples

### Python SDK
```python
import requests

# Analyze contract
response = requests.post('http://localhost:8000/api/v1/analyze', json={
    'contract_code': 'pragma solidity ^0.8.0;\ncontract Test {}',
    'contract_name': 'Test'
})

report_id = response.json()['report_id']

# Get report
report = requests.get(f'http://localhost:8000/api/v1/report/{report_id}')
print(report.json())
```

### JavaScript SDK
```javascript
// Analyze contract
const response = await fetch('http://localhost:8000/api/v1/analyze', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    contract_code: 'pragma solidity ^0.8.0;\ncontract Test {}',
    contract_name: 'Test'
  })
});

const { report_id } = await response.json();

// Get report
const report = await fetch(`http://localhost:8000/api/v1/report/${report_id}`);
const data = await report.json();
console.log(data);
```

## 🔄 WebSocket Support (Future)

### Real-time Updates
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/analysis/{report_id}');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Analysis progress:', data.progress);
  console.log('Status:', data.status);
};
```

## 📝 Changelog

### Version 1.0.0
- Initial API release
- Basic contract analysis
- Report management
- Batch processing
- Health monitoring

### Future Versions
- WebSocket support for real-time updates
- Advanced filtering and search
- Export functionality
- API versioning
- Enhanced security features

---

For more information, visit the [main documentation](README.md) or check the [technical architecture](TECHNICAL_ARCHITECTURE.md).
