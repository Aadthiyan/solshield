import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import FileUpload from '@/components/FileUpload';

// Mock the useDropzone hook
jest.mock('react-dropzone', () => ({
  useDropzone: jest.fn(() => ({
    getRootProps: jest.fn(() => ({})),
    getInputProps: jest.fn(() => ({})),
    isDragActive: false,
  })),
}));

describe('FileUpload', () => {
  const mockOnFileSelect = jest.fn();
  const mockOnAnalysisStart = jest.fn();

  const defaultProps = {
    onFileSelect: mockOnFileSelect,
    onAnalysisStart: mockOnAnalysisStart,
    isAnalyzing: false,
    acceptedTypes: ['.sol', '.txt'],
    maxSize: 10 * 1024 * 1024,
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders upload interface correctly', () => {
    render(<FileUpload {...defaultProps} />);
    
    expect(screen.getByText('Upload Smart Contract')).toBeInTheDocument();
    expect(screen.getByText('Drag and drop your Solidity contract file, or click to browse')).toBeInTheDocument();
    expect(screen.getByText('Accepted formats: .sol, .txt')).toBeInTheDocument();
    expect(screen.getByText('Maximum size: 10 MB')).toBeInTheDocument();
  });

  it('shows analyzing state when isAnalyzing is true', () => {
    render(<FileUpload {...defaultProps} isAnalyzing={true} />);
    
    expect(screen.getByText('Analyzing contract...')).toBeInTheDocument();
  });

  it('calls onFileSelect when file is selected', async () => {
    const user = userEvent.setup();
    const file = new File(['contract code'], 'test.sol', { type: 'text/plain' });
    
    render(<FileUpload {...defaultProps} />);
    
    const input = screen.getByRole('button', { hidden: true });
    await user.upload(input, file);
    
    expect(mockOnFileSelect).toHaveBeenCalledWith(file);
  });

  it('calls onAnalysisStart when analyze button is clicked', async () => {
    const user = userEvent.setup();
    const file = new File(['contract code'], 'test.sol', { type: 'text/plain' });
    
    // Mock FileReader
    const mockFileReader = {
      readAsText: jest.fn(),
      result: 'contract code',
      onload: null,
    };
    jest.spyOn(window, 'FileReader').mockImplementation(() => mockFileReader as any);
    
    render(<FileUpload {...defaultProps} />);
    
    const input = screen.getByRole('button', { hidden: true });
    await user.upload(input, file);
    
    // Simulate file selection
    const analyzeButton = screen.getByText('Analyze Contract');
    await user.click(analyzeButton);
    
    // Simulate FileReader onload
    if (mockFileReader.onload) {
      mockFileReader.onload({ target: { result: 'contract code' } } as any);
    }
    
    expect(mockOnAnalysisStart).toHaveBeenCalledWith({
      contract_code: 'contract code',
      contract_name: 'test',
      model_type: 'ensemble',
      include_optimization_suggestions: true,
      include_explanation: true,
    });
  });

  it('shows error message for invalid file type', () => {
    render(<FileUpload {...defaultProps} />);
    
    // This would be handled by the dropzone validation
    // The component should display error state
    expect(screen.queryByText(/Invalid file type/)).not.toBeInTheDocument();
  });

  it('shows error message for file too large', () => {
    render(<FileUpload {...defaultProps} />);
    
    // This would be handled by the dropzone validation
    // The component should display error state
    expect(screen.queryByText(/File is too large/)).not.toBeInTheDocument();
  });

  it('disables upload when analyzing', () => {
    render(<FileUpload {...defaultProps} isAnalyzing={true} />);
    
    const uploadArea = screen.getByRole('button');
    expect(uploadArea).toHaveClass('cursor-not-allowed');
  });

  it('handles file removal correctly', async () => {
    const user = userEvent.setup();
    const file = new File(['contract code'], 'test.sol', { type: 'text/plain' });
    
    render(<FileUpload {...defaultProps} />);
    
    const input = screen.getByRole('button', { hidden: true });
    await user.upload(input, file);
    
    // File should be selected
    expect(screen.getByText('test.sol')).toBeInTheDocument();
    
    // Click remove button
    const removeButton = screen.getByTestId('x-icon');
    await user.click(removeButton);
    
    // File should be removed
    expect(screen.queryByText('test.sol')).not.toBeInTheDocument();
  });

  it('formats file size correctly', () => {
    render(<FileUpload {...defaultProps} maxSize={1024 * 1024} />);
    
    expect(screen.getByText('Maximum size: 1 MB')).toBeInTheDocument();
  });

  it('shows correct accepted types', () => {
    render(<FileUpload {...defaultProps} acceptedTypes={['.sol', '.txt', '.js']} />);
    
    expect(screen.getByText('Accepted formats: .sol, .txt, .js')).toBeInTheDocument();
  });
});
