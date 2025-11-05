describe('Navigation and User Experience', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.mockApi();
  });

  it('should navigate between upload and dashboard', () => {
    // Start on upload page
    cy.contains('Upload Smart Contract').should('be.visible');
    
    // Navigate to dashboard
    cy.contains('Dashboard').click();
    cy.contains('Dashboard').should('be.visible');
    cy.contains('Smart Contract Vulnerability Analysis Overview').should('be.visible');
    
    // Navigate back to upload
    cy.contains('Upload').click();
    cy.contains('Upload Smart Contract').should('be.visible');
  });

  it('should maintain state during navigation', () => {
    // Upload a file
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.contains('test.sol').should('be.visible');
    
    // Navigate to dashboard
    cy.contains('Dashboard').click();
    cy.contains('Dashboard').should('be.visible');
    
    // Navigate back to upload
    cy.contains('Upload').click();
    
    // File should still be selected
    cy.contains('test.sol').should('be.visible');
  });

  it('should handle responsive design', () => {
    // Test mobile viewport
    cy.viewport(375, 667);
    cy.contains('Upload Smart Contract').should('be.visible');
    
    // Test tablet viewport
    cy.viewport(768, 1024);
    cy.contains('Upload Smart Contract').should('be.visible');
    
    // Test desktop viewport
    cy.viewport(1280, 720);
    cy.contains('Upload Smart Contract').should('be.visible');
  });

  it('should display header and footer correctly', () => {
    // Check header
    cy.contains('Smart Contract Vulnerability Detection').should('be.visible');
    cy.contains('AI-powered security analysis').should('be.visible');
    
    // Check navigation
    cy.contains('Upload').should('be.visible');
    cy.contains('Dashboard').should('be.visible');
    
    // Scroll to footer
    cy.scrollTo('bottom');
    
    // Check footer
    cy.contains('© 2024 Smart Contract Vulnerability Detection').should('be.visible');
    cy.contains('Powered by AI models including CodeBERT and Graph Neural Networks').should('be.visible');
  });

  it('should handle keyboard navigation', () => {
    // Test tab navigation
    cy.get('body').type('{tab}');
    cy.focused().should('have.attr', 'data-testid', 'upload-button');
    
    cy.get('body').type('{tab}');
    cy.focused().should('have.attr', 'data-testid', 'dashboard-button');
  });

  it('should handle loading states', () => {
    // Mock slow API response
    cy.intercept('POST', '/api/v1/analyze', {
      delay: 2000,
      statusCode: 200,
      body: {
        success: true,
        message: 'Analysis completed successfully',
        report_id: 'test-report-id-123',
        estimated_processing_time: 2.5,
      },
    }).as('slowAnalyze');
    
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    // Should show loading state
    cy.contains('Analyzing contract...').should('be.visible');
    cy.get('[data-testid="progress-indicator"]').should('be.visible');
  });

  it('should handle error states gracefully', () => {
    // Mock API error
    cy.intercept('GET', '/api/v1/health', {
      statusCode: 500,
      body: {
        error: 'Service unavailable',
      },
    }).as('healthError');
    
    // Should show error state
    cy.contains('Service unavailable').should('be.visible');
  });

  it('should handle network connectivity issues', () => {
    // Simulate network error
    cy.intercept('POST', '/api/v1/analyze', {
      forceNetworkError: true,
    }).as('networkError');
    
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    // Should show network error
    cy.contains('Network error').should('be.visible');
  });

  it('should persist user preferences', () => {
    // Navigate to dashboard
    cy.contains('Dashboard').click();
    
    // Apply filters
    cy.get('[data-testid="filter-button"]').click();
    cy.get('[data-testid="severity-filter"]').check();
    
    // Navigate away and back
    cy.contains('Upload').click();
    cy.contains('Dashboard').click();
    
    // Filters should be persisted
    cy.get('[data-testid="severity-filter"]').should('be.checked');
  });

  it('should handle browser back/forward navigation', () => {
    // Navigate to dashboard
    cy.contains('Dashboard').click();
    cy.url().should('include', '#dashboard');
    
    // Go back
    cy.go('back');
    cy.url().should('not.include', '#dashboard');
    
    // Go forward
    cy.go('forward');
    cy.url().should('include', '#dashboard');
  });

  it('should handle page refresh', () => {
    // Upload a file
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.contains('test.sol').should('be.visible');
    
    // Refresh page
    cy.reload();
    
    // Should maintain state
    cy.contains('test.sol').should('be.visible');
  });
});
