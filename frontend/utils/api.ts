import axios, { AxiosInstance, AxiosResponse, AxiosError } from 'axios';
import {
  ContractSubmissionRequest,
  ContractSubmissionResponse,
  ReportRetrievalResponse,
  BatchSubmissionRequest,
  BatchSubmissionResponse,
  BatchReportResponse,
  HealthCheckResponse,
  SystemStatusResponse,
  MetricsResponse,
  ApiResponse,
  ApiError,
} from '@/types';
import { API_CONFIG, API_ENDPOINTS, ERROR_MESSAGES } from '@/config/api';

class ApiClient {
  private client: AxiosInstance;
  private baseURL: string;

  constructor(baseURL: string = API_CONFIG.BASE_URL) {
    this.baseURL = baseURL;
    this.client = axios.create({
      baseURL: this.baseURL,
      timeout: API_CONFIG.TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add request timestamp
        (config as any).metadata = { startTime: Date.now() };
        return config;
      },
      (error) => {
        return Promise.reject(this.handleError(error));
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => {
        // Add response time
        const endTime = Date.now();
        const startTime = (response.config as any).metadata?.startTime;
        if (startTime) {
          (response as any).metadata = { responseTime: endTime - startTime };
        }
        return response;
      },
      (error) => {
        return Promise.reject(this.handleError(error));
      }
    );
  }

  private handleError(error: AxiosError): ApiError {
    if (error.response) {
      // Server responded with error status
      return {
        message: (error.response.data as any)?.error || error.message,
        status: error.response.status,
        code: (error.response.data as any)?.code,
        details: error.response.data,
      };
    } else if (error.request) {
      // Request was made but no response received
      return {
        message: ERROR_MESSAGES.NETWORK_ERROR,
        status: 0,
        code: 'NETWORK_ERROR',
      };
    } else {
      // Something else happened
      return {
        message: error.message || ERROR_MESSAGES.UNKNOWN_ERROR,
        code: 'UNKNOWN_ERROR',
      };
    }
  }

  // Health and Status endpoints
  async healthCheck(): Promise<ApiResponse<HealthCheckResponse>> {
    const response = await this.client.get(API_ENDPOINTS.HEALTH);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  async getSystemStatus(): Promise<ApiResponse<SystemStatusResponse>> {
    const response = await this.client.get(API_ENDPOINTS.STATUS);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  async getMetrics(): Promise<ApiResponse<MetricsResponse>> {
    const response = await this.client.get(API_ENDPOINTS.METRICS);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  // Contract analysis endpoints
  async analyzeContract(request: ContractSubmissionRequest): Promise<ApiResponse<ContractSubmissionResponse>> {
    const response = await this.client.post(API_ENDPOINTS.ANALYZE, request);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  async getReport(reportId: string): Promise<ApiResponse<ReportRetrievalResponse>> {
    const response = await this.client.get(`${API_ENDPOINTS.REPORT}/${reportId}`);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  async batchAnalyze(request: BatchSubmissionRequest): Promise<ApiResponse<BatchSubmissionResponse>> {
    const response = await this.client.post(API_ENDPOINTS.BATCH_ANALYZE, request);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  async getBatchReports(batchId: string): Promise<ApiResponse<BatchReportResponse>> {
    const response = await this.client.get(`${API_ENDPOINTS.BATCH_REPORTS}/${batchId}`);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  async listReports(limit: number = 10, offset: number = 0): Promise<ApiResponse<any>> {
    const response = await this.client.get(API_ENDPOINTS.REPORTS, {
      params: { limit, offset },
    });
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  async deleteReport(reportId: string): Promise<ApiResponse<any>> {
    const response = await this.client.delete(`${API_ENDPOINTS.REPORT}/${reportId}`);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
      headers: response.headers as Record<string, string>,
    };
  }

  // Utility methods
  async checkConnection(): Promise<boolean> {
    try {
      await this.healthCheck();
      return true;
    } catch (error) {
      return false;
    }
  }

  getBaseURL(): string {
    return this.baseURL;
  }

  setBaseURL(baseURL: string): void {
    this.baseURL = baseURL;
    this.client.defaults.baseURL = baseURL;
  }
}

// Create singleton instance with error handling
let apiClient: ApiClient;

try {
  apiClient = new ApiClient();
} catch (error) {
  console.error('Failed to initialize API client:', error);
  // Create a fallback client with default settings
  apiClient = new ApiClient('http://localhost:8000');
}

export default apiClient;

// Export individual methods for convenience with error handling
export const healthCheck = (...args: any[]) => {
  try {
    return apiClient.healthCheck(...args);
  } catch (error) {
    console.error('Health check error:', error);
    throw error;
  }
};

export const getSystemStatus = (...args: any[]) => {
  try {
    return apiClient.getSystemStatus(...args);
  } catch (error) {
    console.error('Get system status error:', error);
    throw error;
  }
};

export const getMetrics = (...args: any[]) => {
  try {
    return apiClient.getMetrics(...args);
  } catch (error) {
    console.error('Get metrics error:', error);
    throw error;
  }
};

export const analyzeContract = (...args: any[]) => {
  try {
    return apiClient.analyzeContract(...args);
  } catch (error) {
    console.error('Analyze contract error:', error);
    throw error;
  }
};

export const getReport = (...args: any[]) => {
  try {
    return apiClient.getReport(...args);
  } catch (error) {
    console.error('Get report error:', error);
    throw error;
  }
};

export const batchAnalyze = (...args: any[]) => {
  try {
    return apiClient.batchAnalyze(...args);
  } catch (error) {
    console.error('Batch analyze error:', error);
    throw error;
  }
};

export const getBatchReports = (...args: any[]) => {
  try {
    return apiClient.getBatchReports(...args);
  } catch (error) {
    console.error('Get batch reports error:', error);
    throw error;
  }
};

export const listReports = (...args: any[]) => {
  try {
    return apiClient.listReports(...args);
  } catch (error) {
    console.error('List reports error:', error);
    throw error;
  }
};

export const deleteReport = (...args: any[]) => {
  try {
    return apiClient.deleteReport(...args);
  } catch (error) {
    console.error('Delete report error:', error);
    throw error;
  }
};

export const checkConnection = (...args: any[]) => {
  try {
    return apiClient.checkConnection(...args);
  } catch (error) {
    console.error('Check connection error:', error);
    throw error;
  }
};

export const getBaseURL = (...args: any[]) => {
  try {
    return apiClient.getBaseURL(...args);
  } catch (error) {
    console.error('Get base URL error:', error);
    throw error;
  }
};

export const setBaseURL = (...args: any[]) => {
  try {
    return apiClient.setBaseURL(...args);
  } catch (error) {
    console.error('Set base URL error:', error);
    throw error;
  }
};
