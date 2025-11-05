describe('File Upload and Analysis', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.mockApi();
  });

  it('should display upload interface correctly', () => {
    cy.get('[data-testid="upload-interface"]').should('be.visible');
    cy.contains('Upload Smart Contract').should('be.visible');
    cy.contains('Drag and drop your Solidity contract file, or click to browse').should('be.visible');
    cy.contains('Accepted formats: .sol, .txt').should('be.visible');
    cy.contains('Maximum size: 10 MB').should('be.visible');
  });

  it('should handle file upload and analysis', () => {
    // Upload a test file
    cy.uploadFile('input[type="file"]', 'test.sol', 'contract Test { function test() public {} }');
    
    // Check that file is selected
    cy.contains('test.sol').should('be.visible');
    cy.contains('Analyze Contract').should('be.visible');
    
    // Click analyze button
    cy.get('[data-testid="analyze-button"]').click();
    
    // Wait for analysis to start
    cy.contains('Analyzing contract...').should('be.visible');
    
    // Wait for API call
    cy.waitForApi('@analyzeContract');
    
    // Wait for report retrieval
    cy.waitForApi('@getReport');
    
    // Check that report is displayed
    cy.contains('Vulnerability Report').should('be.visible');
    cy.contains('TestContract').should('be.visible');
  });

  it('should show progress indicator during analysis', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    // Check progress indicator
    cy.get('[data-testid="progress-indicator"]').should('be.visible');
    cy.contains('Analysis Progress').should('be.visible');
    cy.contains('Processing...').should('be.visible');
  });

  it('should handle file removal', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.contains('test.sol').should('be.visible');
    
    // Click remove button
    cy.get('[data-testid="remove-file-button"]').click();
    
    // File should be removed
    cy.contains('test.sol').should('not.exist');
    cy.contains('Upload Smart Contract').should('be.visible');
  });

  it('should validate file types', () => {
    // Try to upload invalid file type
    cy.uploadFile('input[type="file"]', 'test.txt', 'This is not Solidity code');
    
    // Should show error or validation message
    cy.contains('Invalid file type').should('be.visible');
  });

  it('should handle large files', () => {
    // Create a large file content
    const largeContent = 'contract Test { ' + 'function test() public {} '.repeat(1000) + '}';
    
    cy.uploadFile('input[type="file"]', 'large.sol', largeContent);
    
    // Should handle large files gracefully
    cy.contains('large.sol').should('be.visible');
  });

  it('should show error for failed analysis', () => {
    // Mock API error
    cy.intercept('POST', '/api/v1/analyze', {
      statusCode: 500,
      body: {
        error: 'Analysis failed',
        detail: 'Internal server error',
      },
    }).as('analyzeError');
    
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.wait('@analyzeError');
    
    // Should show error message
    cy.contains('Analysis failed').should('be.visible');
  });
});
