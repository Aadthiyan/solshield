import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import HomePage from '@/pages/index';

// Mock the store
const mockStore = {
  reports: [],
  currentReport: null,
  isAnalyzing: false,
  analysisProgress: {
    isAnalyzing: false,
    progress: 0,
    currentStep: '',
    estimatedTime: 0,
  },
  dashboardStats: null,
  error: null,
  addReport: jest.fn(),
  setCurrentReport: jest.fn(),
  setAnalyzing: jest.fn(),
  setAnalysisProgress: jest.fn(),
  setDashboardStats: jest.fn(),
  setError: jest.fn(),
  clearError: jest.fn(),
};

jest.mock('@/utils/store', () => ({
  useAppStore: () => mockStore,
}));

// Mock the API
const mockAnalyzeContract = jest.fn();
const mockGetReport = jest.fn();

jest.mock('@/utils/api', () => ({
  analyzeContract: mockAnalyzeContract,
  getReport: mockGetReport,
}));

describe('HomePage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders homepage correctly', () => {
    render(<HomePage />);
    
    expect(screen.getByText('Smart Contract Vulnerability Detection')).toBeInTheDocument();
    expect(screen.getByText('AI-powered security analysis')).toBeInTheDocument();
    expect(screen.getByText('Upload Smart Contract')).toBeInTheDocument();
  });

  it('shows upload interface by default', () => {
    render(<HomePage />);
    
    expect(screen.getByText('Upload Smart Contract')).toBeInTheDocument();
    expect(screen.getByText('Drag and drop your Solidity contract file, or click to browse')).toBeInTheDocument();
  });

  it('switches to dashboard view when dashboard button is clicked', async () => {
    const user = userEvent.setup();
    render(<HomePage />);
    
    const dashboardButton = screen.getByText('Dashboard');
    await user.click(dashboardButton);
    
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('shows progress indicator when analyzing', () => {
    mockStore.isAnalyzing = true;
    mockStore.analysisProgress = {
      isAnalyzing: true,
      progress: 50,
      currentStep: 'Processing...',
      estimatedTime: 30,
    };
    
    render(<HomePage />);
    
    expect(screen.getByText('Analysis Progress')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('Processing...')).toBeInTheDocument();
  });

  it('shows error message when error occurs', () => {
    mockStore.error = 'Analysis failed';
    
    render(<HomePage />);
    
    expect(screen.getByText('Analysis failed')).toBeInTheDocument();
  });

  it('handles file upload and analysis', async () => {
    const user = userEvent.setup();
    const file = new File(['contract code'], 'test.sol', { type: 'text/plain' });
    
    // Mock successful API responses
    mockAnalyzeContract.mockResolvedValue({
      data: {
        success: true,
        report_id: 'test-report-id',
        message: 'Analysis started',
      },
    });
    
    mockGetReport.mockResolvedValue({
      data: {
        success: true,
        report: {
          report_id: 'test-report-id',
          contract_name: 'test',
          is_vulnerable: true,
          risk_score: 7.5,
          vulnerabilities: [],
          optimization_suggestions: [],
          model_predictions: [],
          processing_time: 2.5,
          timestamp: new Date().toISOString(),
          contract_hash: 'test-hash',
          overall_confidence: 0.85,
        },
      },
    });
    
    // Mock FileReader
    const mockFileReader = {
      readAsText: jest.fn(),
      result: 'contract code',
      onload: null,
    };
    jest.spyOn(window, 'FileReader').mockImplementation(() => mockFileReader as any);
    
    render(<HomePage />);
    
    const input = screen.getByRole('button', { hidden: true });
    await user.upload(input, file);
    
    // Wait for file to be processed
    await waitFor(() => {
      expect(screen.getByText('test.sol')).toBeInTheDocument();
    });
    
    // Click analyze button
    const analyzeButton = screen.getByText('Analyze Contract');
    await user.click(analyzeButton);
    
    // Simulate FileReader onload
    if (mockFileReader.onload) {
      mockFileReader.onload({ target: { result: 'contract code' } } as any);
    }
    
    // Wait for analysis to complete
    await waitFor(() => {
      expect(mockAnalyzeContract).toHaveBeenCalled();
    });
  });

  it('shows dashboard stats correctly', () => {
    mockStore.dashboardStats = {
      totalAnalyses: 10,
      vulnerableContracts: 3,
      averageRiskScore: 5.5,
      topVulnerabilityTypes: [
        { type: 'reentrancy', count: 2 },
        { type: 'integer_overflow', count: 1 },
      ],
      recentAnalyses: [],
    };
    
    render(<HomePage />);
    
    // Switch to dashboard view
    const dashboardButton = screen.getByText('Dashboard');
    fireEvent.click(dashboardButton);
    
    expect(screen.getByText('10')).toBeInTheDocument(); // Total analyses
    expect(screen.getByText('3')).toBeInTheDocument(); // Vulnerable contracts
    expect(screen.getByText('5.5')).toBeInTheDocument(); // Average risk score
  });

  it('handles navigation between views', async () => {
    const user = userEvent.setup();
    render(<HomePage />);
    
    // Start with upload view
    expect(screen.getByText('Upload Smart Contract')).toBeInTheDocument();
    
    // Switch to dashboard
    const dashboardButton = screen.getByText('Dashboard');
    await user.click(dashboardButton);
    
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
    
    // Switch back to upload
    const uploadButton = screen.getByText('Upload');
    await user.click(uploadButton);
    
    expect(screen.getByText('Upload Smart Contract')).toBeInTheDocument();
  });

  it('displays footer correctly', () => {
    render(<HomePage />);
    
    expect(screen.getByText('© 2024 Smart Contract Vulnerability Detection. All rights reserved.')).toBeInTheDocument();
    expect(screen.getByText('Powered by AI models including CodeBERT and Graph Neural Networks')).toBeInTheDocument();
  });

  it('handles API errors gracefully', async () => {
    const user = userEvent.setup();
    
    // Mock API error
    mockAnalyzeContract.mockRejectedValue(new Error('API Error'));
    
    render(<HomePage />);
    
    const file = new File(['contract code'], 'test.sol', { type: 'text/plain' });
    
    // Mock FileReader
    const mockFileReader = {
      readAsText: jest.fn(),
      result: 'contract code',
      onload: null,
    };
    jest.spyOn(window, 'FileReader').mockImplementation(() => mockFileReader as any);
    
    const input = screen.getByRole('button', { hidden: true });
    await user.upload(input, file);
    
    const analyzeButton = screen.getByText('Analyze Contract');
    await user.click(analyzeButton);
    
    // Simulate FileReader onload
    if (mockFileReader.onload) {
      mockFileReader.onload({ target: { result: 'contract code' } } as any);
    }
    
    await waitFor(() => {
      expect(mockStore.setError).toHaveBeenCalled();
    });
  });
});
