# Smart Contract Vulnerability Detection dApp - User Guide

## Overview

The Smart Contract Vulnerability Detection dApp is an AI-powered platform that automatically analyzes smart contracts for security vulnerabilities and provides optimization suggestions. The system combines state-of-the-art machine learning models with static analysis tools to deliver comprehensive security assessments.

## Features

- **AI-Powered Analysis**: Uses CodeBERT and Graph Neural Networks for intelligent vulnerability detection
- **Static Analysis Integration**: Incorporates Slither and Mythril for comprehensive coverage
- **Real-time Processing**: Fast analysis with progress indicators
- **Detailed Reports**: Comprehensive vulnerability reports with severity ratings and recommendations
- **Gas Optimization**: Suggests optimizations to reduce gas consumption
- **Batch Processing**: Analyze multiple contracts simultaneously
- **Web Interface**: User-friendly React-based frontend
- **API Access**: RESTful API for programmatic access

## Getting Started

### Prerequisites

- Node.js 16+ and npm
- Python 3.8+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd smart-contract-vulnerability-detection
   ```

2. **Install backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Quick Start

1. **Start the backend API**:
   ```bash
   python -m api.main
   ```

2. **Start the frontend**:
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the application**:
   Open your browser and navigate to `http://localhost:3000`

## Using the Web Interface

### Uploading Contracts

1. **Navigate to the Dashboard**: The main interface loads automatically
2. **Upload Contract**: 
   - Drag and drop a `.sol` file onto the upload area, or
   - Click "Choose File" to browse and select a contract file
3. **Monitor Progress**: Watch the real-time progress indicator
4. **View Results**: Once analysis is complete, view the detailed report

### Understanding Reports

#### Vulnerability Cards
Each detected vulnerability is displayed as a card showing:
- **Type**: The specific vulnerability type (e.g., Reentrancy, Integer Overflow)
- **Severity**: Risk level (Critical, High, Medium, Low)
- **Confidence**: AI model confidence score (0-100%)
- **Location**: Where in the code the vulnerability was found
- **Description**: Detailed explanation of the issue
- **Recommendations**: Specific steps to fix the vulnerability

#### Optimization Suggestions
Gas optimization suggestions include:
- **Current Gas Usage**: Estimated gas consumption
- **Optimized Gas Usage**: Potential savings
- **Savings Percentage**: Percentage reduction achievable
- **Implementation**: Code changes needed

### Dashboard Features

- **Recent Analyses**: View your last 10 contract analyses
- **Statistics**: Overview of vulnerability types and severity distribution
- **Export Reports**: Download analysis results as JSON or PDF
- **Batch Analysis**: Upload multiple contracts for analysis

## API Usage

### Authentication

The API uses API keys for authentication. Include your API key in the request headers:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \
     -H "Content-Type: application/json" \
     -X POST http://localhost:8000/analyze \
     -d '{"contract_code": "contract SolidityContract { ... }"}'
```

### Endpoints

#### Analyze Contract
```http
POST /analyze
Content-Type: application/json

{
  "contract_code": "pragma solidity ^0.8.0; contract MyContract { ... }"
}
```

**Response**:
```json
{
  "analysis_id": "uuid",
  "vulnerabilities": [
    {
      "type": "reentrancy",
      "severity": "high",
      "confidence": 0.95,
      "description": "External call before state update",
      "location": "withdraw function",
      "line_number": 15
    }
  ],
  "optimizations": [
    {
      "type": "gas_optimization",
      "description": "Use unchecked arithmetic",
      "potential_savings": 2000,
      "implementation": "Use unchecked { ... }"
    }
  ],
  "risk_score": 0.8,
  "analysis_time": 2.5
}
```

#### Get Analysis Report
```http
GET /reports/{analysis_id}
```

#### Batch Analysis
```http
POST /analyze/batch
Content-Type: application/json

{
  "contracts": [
    {"name": "contract1.sol", "code": "..."},
    {"name": "contract2.sol", "code": "..."}
  ]
}
```

### Error Handling

The API returns standard HTTP status codes:
- `200`: Success
- `400`: Bad Request (invalid input)
- `401`: Unauthorized (missing/invalid API key)
- `429`: Rate Limited
- `500`: Internal Server Error

Error responses include details:
```json
{
  "error": "Invalid contract code",
  "code": "INVALID_INPUT",
  "details": "Contract compilation failed"
}
```

## Advanced Features

### Custom Model Training

To train custom models with your own data:

1. **Prepare Training Data**:
   ```bash
   python scripts/prepare_training_data.py --input data/raw --output data/processed
   ```

2. **Train Models**:
   ```bash
   python training/train_models.py --config config/training_config.yaml
   ```

3. **Evaluate Performance**:
   ```bash
   python evaluation/model_comparison.py --models models/
   ```

### Static Analysis Integration

The system integrates with popular static analysis tools:

- **Slither**: Comprehensive static analysis
- **Mythril**: Symbolic execution analysis

To run static analysis only:
```bash
python benchmarking/static_analysis.py --tool slither --contract contract.sol
```

### Performance Benchmarking

Run comprehensive performance tests:

```bash
python benchmarking/performance_benchmark.py --config config/benchmark_config.yaml
```

This generates:
- Accuracy validation reports
- Performance metrics
- Comparison with static analysis tools
- Recommendations for improvement

## Deployment

### Docker Deployment

1. **Build Images**:
   ```bash
   docker-compose build
   ```

2. **Start Services**:
   ```bash
   docker-compose up -d
   ```

3. **Access Application**:
   - Frontend: `http://localhost:3000`
   - API: `http://localhost:8000`

### QIE Testnet Deployment

Deploy the dApp to QIE testnet:

1. **Configure QIE Settings**:
   ```bash
   cp deployment/qie_config.example.json deployment/qie_config.json
   # Edit with your QIE network details
   ```

2. **Deploy Contracts**:
   ```bash
   python deployment/qie_deployment.py --deploy
   ```

3. **Verify Deployment**:
   ```bash
   python deployment/qie_deployment.py --verify
   ```

## Troubleshooting

### Common Issues

#### Backend Won't Start
- Check Python version (3.8+ required)
- Verify all dependencies are installed: `pip install -r requirements.txt`
- Check port 8000 is available
- Review logs for specific error messages

#### Frontend Build Errors
- Ensure Node.js 16+ is installed
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`
- Check for TypeScript errors: `npm run type-check`

#### Model Loading Issues
- Verify model files exist in `models/` directory
- Check model file permissions
- Ensure sufficient memory available
- Review model loading logs

#### API Rate Limiting
- Implement exponential backoff in your client
- Use batch endpoints for multiple contracts
- Consider upgrading to higher rate limits

### Performance Optimization

#### For Large Contracts
- Split large contracts into smaller modules
- Use batch processing for multiple contracts
- Consider using the API's async endpoints

#### For High Volume
- Deploy multiple API instances behind a load balancer
- Use Redis for caching analysis results
- Implement database connection pooling

### Getting Help

1. **Check Documentation**: Review this guide and API documentation
2. **Search Issues**: Look through existing GitHub issues
3. **Create Issue**: Submit a detailed bug report with:
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages and logs
   - Environment details

## Security Considerations

### API Key Management
- Store API keys securely
- Use environment variables, not hardcoded values
- Rotate keys regularly
- Implement proper access controls

### Contract Privacy
- Contracts are processed in memory only
- No persistent storage of contract code
- Analysis results may be logged (without code)
- Use HTTPS in production

### Rate Limiting
- Implement client-side rate limiting
- Respect API rate limits
- Use exponential backoff for retries

## Best Practices

### Contract Analysis
1. **Test Thoroughly**: Always test contracts after implementing fixes
2. **Review Recommendations**: Understand the reasoning behind suggestions
3. **Multiple Tools**: Use both AI and static analysis for comprehensive coverage
4. **Regular Updates**: Keep models and tools updated

### Integration
1. **Error Handling**: Implement robust error handling in your applications
2. **Caching**: Cache analysis results when appropriate
3. **Monitoring**: Monitor API usage and performance
4. **Documentation**: Document your integration approach

### Development
1. **Version Control**: Use version control for all code changes
2. **Testing**: Write comprehensive tests for your integrations
3. **Code Review**: Implement code review processes
4. **Security**: Follow security best practices

## Changelog

### Version 1.0.0
- Initial release
- AI-powered vulnerability detection
- Web interface
- API access
- Static analysis integration
- QIE testnet deployment

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Support

For support and questions:
- Email: support@example.com
- Documentation: https://docs.example.com
- GitHub: https://github.com/example/smart-contract-vulnerability-detection
