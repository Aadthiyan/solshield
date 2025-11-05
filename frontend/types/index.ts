// API Types
export interface ContractSubmissionRequest {
  contract_code: string;
  contract_name?: string;
  model_type: 'codebert' | 'gnn' | 'ensemble';
  include_optimization_suggestions: boolean;
  include_explanation: boolean;
}

export interface ContractSubmissionResponse {
  success: boolean;
  message: string;
  report_id?: string;
  estimated_processing_time?: number;
}

export interface VulnerabilityType {
  type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  confidence: number;
  description: string;
  location?: string;
  explanation?: string;
  recommendation?: string;
  references?: string[];
}

export interface OptimizationSuggestion {
  type: string;
  description: string;
  potential_savings?: string;
  implementation?: string;
  priority: 'high' | 'medium' | 'low';
}

export interface ModelPrediction {
  model_type: string;
  is_vulnerable: boolean;
  confidence: number;
  vulnerability_types: string[];
  processing_time: number;
}

export interface VulnerabilityReport {
  report_id: string;
  timestamp: string;
  contract_name?: string;
  contract_hash: string;
  is_vulnerable: boolean;
  overall_confidence: number;
  risk_score: number;
  vulnerabilities: VulnerabilityType[];
  optimization_suggestions: OptimizationSuggestion[];
  model_predictions: ModelPrediction[];
  processing_time: number;
  model_versions: Record<string, string>;
}

export interface ReportRetrievalResponse {
  success: boolean;
  message: string;
  report?: VulnerabilityReport;
  error?: string;
}

export interface BatchSubmissionRequest {
  contracts: ContractSubmissionRequest[];
  batch_id?: string;
}

export interface BatchSubmissionResponse {
  success: boolean;
  message: string;
  batch_id: string;
  contract_ids: string[];
  estimated_processing_time: number;
}

export interface BatchReportResponse {
  success: boolean;
  message: string;
  batch_id: string;
  reports: VulnerabilityReport[];
  processing_status: Record<string, string>;
  total_processing_time: number;
}

export interface HealthCheckResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: string;
  version: string;
  models_loaded: Record<string, boolean>;
  uptime: number;
  memory_usage: {
    total: number;
    available: number;
    percent: number;
    used: number;
  };
}

export interface SystemStatusResponse {
  status: string;
  timestamp: string;
  models: ModelInfo[];
  system_metrics: Record<string, any>;
  active_requests: number;
  queue_size: number;
}

export interface ModelInfo {
  model_type: string;
  version: string;
  is_loaded: boolean;
  load_time?: number;
  last_used?: number;
  performance_metrics?: Record<string, number>;
}

export interface MetricsResponse {
  total_requests: number;
  successful_requests: number;
  failed_requests: number;
  average_processing_time: number;
  model_usage_stats: Record<string, number>;
  vulnerability_type_stats: Record<string, number>;
  time_range: {
    start: string;
    end: string;
  };
}

export interface ErrorResponse {
  error: string;
  detail?: string;
  timestamp: string;
  request_id?: string;
}

// UI Types
export interface UploadProgress {
  file: File;
  progress: number;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  error?: string;
  report_id?: string;
}

export interface AnalysisState {
  isAnalyzing: boolean;
  progress: number;
  currentStep: string;
  estimatedTime: number;
  startTime?: number;
}

export interface DashboardStats {
  totalAnalyses: number;
  vulnerableContracts: number;
  averageRiskScore: number;
  topVulnerabilityTypes: Array<{
    type: string;
    count: number;
  }>;
  recentAnalyses: VulnerabilityReport[];
}

export interface FilterOptions {
  vulnerabilityTypes: string[];
  severityLevels: string[];
  dateRange: {
    start: Date;
    end: Date;
  };
  riskScoreRange: {
    min: number;
    max: number;
  };
}

export interface SortOptions {
  field: 'timestamp' | 'risk_score' | 'vulnerability_count' | 'contract_name';
  direction: 'asc' | 'desc';
}

// Component Props
export interface FileUploadProps {
  onFileSelect: (file: File) => void;
  onAnalysisStart: (contract: ContractSubmissionRequest) => void;
  isAnalyzing: boolean;
  acceptedTypes?: string[];
  maxSize?: number;
}

export interface ReportDisplayProps {
  report: VulnerabilityReport;
  isLoading?: boolean;
  onClose?: () => void;
  showDetails?: boolean;
}

export interface VulnerabilityCardProps {
  vulnerability: VulnerabilityType;
  index: number;
  onExpand?: (index: number) => void;
  isExpanded?: boolean;
}

export interface OptimizationCardProps {
  suggestion: OptimizationSuggestion;
  index: number;
  onExpand?: (index: number) => void;
  isExpanded?: boolean;
  onImplement?: (suggestion: OptimizationSuggestion) => void;
}

export interface DashboardProps {
  stats: DashboardStats;
  reports: VulnerabilityReport[];
  onReportSelect: (report: VulnerabilityReport) => void;
  onRefresh: () => void;
  isLoading?: boolean;
}

export interface ProgressIndicatorProps {
  progress: number;
  status: string;
  estimatedTime?: number;
  isVisible: boolean;
}

// API Client Types
export interface ApiClientConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  retryDelay: number;
}

export interface ApiResponse<T> {
  data: T;
  status: number;
  statusText: string;
  headers: Record<string, string>;
}

export interface ApiError {
  message: string;
  status?: number;
  code?: string;
  details?: any;
}

// Store Types
export interface AppState {
  reports: VulnerabilityReport[];
  currentReport: VulnerabilityReport | null;
  isAnalyzing: boolean;
  analysisProgress: AnalysisState;
  dashboardStats: DashboardStats | null;
  filters: FilterOptions;
  sortOptions: SortOptions;
  error: string | null;
  loading: boolean;
}

export interface AppActions {
  setReports: (reports: VulnerabilityReport[]) => void;
  addReport: (report: VulnerabilityReport) => void;
  setCurrentReport: (report: VulnerabilityReport | null) => void;
  setAnalyzing: (isAnalyzing: boolean) => void;
  setAnalysisProgress: (progress: AnalysisState) => void;
  setDashboardStats: (stats: DashboardStats) => void;
  setFilters: (filters: FilterOptions) => void;
  setSortOptions: (options: SortOptions) => void;
  setError: (error: string | null) => void;
  setLoading: (loading: boolean) => void;
  clearError: () => void;
  reset: () => void;
  getFilteredReports: () => VulnerabilityReport[];
  getVulnerabilityStats: () => Array<{ type: string; count: number }>;
  getSeverityStats: () => Array<{ severity: string; count: number }>;
  getRecentReports: (limit?: number) => VulnerabilityReport[];
  getHighRiskReports: () => VulnerabilityReport[];
  getVulnerableReports: () => VulnerabilityReport[];
}

// Utility Types
export type SeverityLevel = 'low' | 'medium' | 'high' | 'critical';
export type ModelType = 'codebert' | 'gnn' | 'ensemble';
export type AnalysisStatus = 'idle' | 'uploading' | 'processing' | 'completed' | 'error';

export interface ThemeConfig {
  colors: {
    primary: string;
    secondary: string;
    success: string;
    warning: string;
    error: string;
    critical: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  typography: {
    fontFamily: string;
    fontSize: {
      xs: string;
      sm: string;
      base: string;
      lg: string;
      xl: string;
      '2xl': string;
      '3xl': string;
    };
  };
}

export interface NotificationConfig {
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}
