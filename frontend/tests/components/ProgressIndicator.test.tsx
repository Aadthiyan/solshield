import React from 'react';
import { render, screen } from '@testing-library/react';
import ProgressIndicator from '@/components/ProgressIndicator';

describe('ProgressIndicator', () => {
  const defaultProps = {
    progress: 50,
    status: 'Processing...',
    estimatedTime: 30,
    isVisible: true,
  };

  it('renders progress indicator when visible', () => {
    render(<ProgressIndicator {...defaultProps} />);
    
    expect(screen.getByText('Analysis Progress')).toBeInTheDocument();
    expect(screen.getByText('50%')).toBeInTheDocument();
    expect(screen.getByText('Processing...')).toBeInTheDocument();
    expect(screen.getByText('Estimated time remaining: 30s')).toBeInTheDocument();
  });

  it('does not render when not visible', () => {
    render(<ProgressIndicator {...defaultProps} isVisible={false} />);
    
    expect(screen.queryByText('Analysis Progress')).not.toBeInTheDocument();
  });

  it('shows completion message when progress is 100%', () => {
    render(<ProgressIndicator {...defaultProps} progress={100} />);
    
    expect(screen.getByText('Analysis Complete!')).toBeInTheDocument();
    expect(screen.getByText('Your vulnerability report is ready.')).toBeInTheDocument();
  });

  it('shows error message when status contains error', () => {
    render(<ProgressIndicator {...defaultProps} status="Error occurred" />);
    
    expect(screen.getByText('Analysis Failed')).toBeInTheDocument();
    expect(screen.getByText('Error occurred')).toBeInTheDocument();
  });

  it('displays progress steps correctly', () => {
    render(<ProgressIndicator {...defaultProps} progress={60} />);
    
    expect(screen.getByText('File Upload')).toBeInTheDocument();
    expect(screen.getByText('Code Analysis')).toBeInTheDocument();
    expect(screen.getByText('Vulnerability Detection')).toBeInTheDocument();
    expect(screen.getByText('Report Generation')).toBeInTheDocument();
  });

  it('formats time correctly for seconds', () => {
    render(<ProgressIndicator {...defaultProps} estimatedTime={45} />);
    
    expect(screen.getByText('Estimated time remaining: 45s')).toBeInTheDocument();
  });

  it('formats time correctly for minutes and seconds', () => {
    render(<ProgressIndicator {...defaultProps} estimatedTime={90} />);
    
    expect(screen.getByText('Estimated time remaining: 1m 30s')).toBeInTheDocument();
  });

  it('shows correct progress percentage', () => {
    render(<ProgressIndicator {...defaultProps} progress={75} />);
    
    expect(screen.getByText('75%')).toBeInTheDocument();
  });

  it('handles zero estimated time', () => {
    render(<ProgressIndicator {...defaultProps} estimatedTime={0} />);
    
    expect(screen.queryByText(/Estimated time remaining/)).not.toBeInTheDocument();
  });

  it('shows progress bar with correct width', () => {
    render(<ProgressIndicator {...defaultProps} progress={80} />);
    
    const progressBar = screen.getByRole('progressbar', { hidden: true });
    expect(progressBar).toHaveStyle('width: 80%');
  });

  it('displays status text correctly', () => {
    render(<ProgressIndicator {...defaultProps} status="Running vulnerability detection..." />);
    
    expect(screen.getByText('Running vulnerability detection...')).toBeInTheDocument();
  });
});
