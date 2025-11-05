// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000,
};

// API Endpoints
export const API_ENDPOINTS = {
  HEALTH: '/health',
  STATUS: '/api/v1/status',
  METRICS: '/api/v1/metrics',
  ANALYZE: '/api/v1/analyze',
  REPORT: '/api/v1/report',
  BATCH_ANALYZE: '/api/v1/analyze/batch',
  BATCH_REPORTS: '/api/v1/batch',
  REPORTS: '/api/v1/reports',
};

// Error Messages
export const ERROR_MESSAGES = {
  NETWORK_ERROR: 'Network error - unable to connect to the API',
  TIMEOUT_ERROR: 'Request timeout - the API is taking too long to respond',
  SERVER_ERROR: 'Server error - the API encountered an internal error',
  VALIDATION_ERROR: 'Validation error - please check your input',
  UNKNOWN_ERROR: 'Unknown error occurred',
};

// Success Messages
export const SUCCESS_MESSAGES = {
  ANALYSIS_STARTED: 'Analysis started successfully',
  ANALYSIS_COMPLETED: 'Analysis completed successfully',
  REPORT_RETRIEVED: 'Report retrieved successfully',
  CONNECTION_ESTABLISHED: 'Connection to API established',
};
