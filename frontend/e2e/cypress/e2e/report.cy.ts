describe('Report Display and Interaction', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.mockApi();
  });

  it('should display vulnerability report correctly', () => {
    // Upload and analyze a contract
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Check report header
    cy.contains('Vulnerability Report').should('be.visible');
    cy.contains('TestContract').should('be.visible');
    
    // Check summary cards
    cy.contains('Risk Score').should('be.visible');
    cy.contains('7.5').should('be.visible');
    cy.contains('Vulnerabilities').should('be.visible');
    cy.contains('Confidence').should('be.visible');
  });

  it('should display vulnerability details', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Check vulnerability section
    cy.contains('Vulnerabilities (1)').should('be.visible');
    cy.contains('reentrancy').should('be.visible');
    cy.contains('Reentrancy vulnerability detected').should('be.visible');
    cy.contains('high').should('be.visible');
  });

  it('should expand vulnerability details', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Click on vulnerability to expand
    cy.get('[data-testid="vulnerability-card"]').first().click();
    
    // Check expanded content
    cy.contains('Technical Explanation').should('be.visible');
    cy.contains('External call before state change').should('be.visible');
    cy.contains('Recommendation').should('be.visible');
    cy.contains('Use checks-effects-interactions pattern').should('be.visible');
  });

  it('should display optimization suggestions', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Check optimization section
    cy.contains('Optimization Suggestions (1)').should('be.visible');
    cy.contains('storage_optimization').should('be.visible');
    cy.contains('Consider using smaller data types').should('be.visible');
  });

  it('should expand optimization details', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Click on optimization to expand
    cy.get('[data-testid="optimization-card"]').first().click();
    
    // Check expanded content
    cy.contains('Implementation').should('be.visible');
    cy.contains('Use uint128 instead of uint256').should('be.visible');
    cy.contains('Potential Savings').should('be.visible');
    cy.contains('10-30%').should('be.visible');
  });

  it('should display model predictions', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Check model predictions section
    cy.contains('Model Predictions').should('be.visible');
    cy.contains('codebert').should('be.visible');
    cy.contains('gnn').should('be.visible');
    cy.contains('Vulnerable').should('be.visible');
  });

  it('should handle copy functionality', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Click copy button
    cy.get('[data-testid="copy-report-button"]').click();
    
    // Should show copy confirmation
    cy.contains('Copied!').should('be.visible');
  });

  it('should navigate back to dashboard', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Click back button
    cy.contains('Back to Dashboard').click();
    
    // Should return to dashboard
    cy.contains('Dashboard').should('be.visible');
  });

  it('should display report metadata', () => {
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@getReport');
    
    // Check report footer
    cy.contains('Report ID: test-report-id-123').should('be.visible');
    cy.contains('Processing time: 2.5s').should('be.visible');
  });

  it('should handle empty report', () => {
    // Mock empty report
    cy.intercept('GET', '/api/v1/report/test-report-id-123', {
      statusCode: 200,
      body: {
        success: true,
        message: 'Report retrieved successfully',
        report: {
          report_id: 'test-report-id-123',
          timestamp: new Date().toISOString(),
          contract_name: 'TestContract',
          contract_hash: 'test-hash-123',
          is_vulnerable: false,
          overall_confidence: 0.95,
          risk_score: 2.0,
          vulnerabilities: [],
          optimization_suggestions: [],
          model_predictions: [],
          processing_time: 1.5,
          model_versions: {},
        },
      },
    }).as('emptyReport');
    
    cy.uploadFile('input[type="file"]', 'test.sol');
    cy.get('[data-testid="analyze-button"]').click();
    
    cy.waitForApi('@analyzeContract');
    cy.waitForApi('@emptyReport');
    
    // Should show no vulnerabilities
    cy.contains('No vulnerabilities found').should('be.visible');
    cy.contains('Risk Score').should('be.visible');
    cy.contains('2.0').should('be.visible');
  });
});
