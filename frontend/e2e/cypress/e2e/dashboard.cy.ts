describe('Dashboard and Reports', () => {
  beforeEach(() => {
    cy.visit('/');
    cy.mockApi();
  });

  it('should display dashboard correctly', () => {
    // Navigate to dashboard
    cy.contains('Dashboard').click();
    
    // Check dashboard elements
    cy.contains('Dashboard').should('be.visible');
    cy.contains('Smart Contract Vulnerability Analysis Overview').should('be.visible');
    
    // Check stats cards
    cy.contains('Total Analyses').should('be.visible');
    cy.contains('Vulnerable Contracts').should('be.visible');
    cy.contains('Average Risk Score').should('be.visible');
    cy.contains('Recent Analyses').should('be.visible');
  });

  it('should display charts and visualizations', () => {
    cy.contains('Dashboard').click();
    
    // Check for charts
    cy.get('[data-testid="bar-chart"]').should('be.visible');
    cy.get('[data-testid="pie-chart"]').should('be.visible');
    cy.get('[data-testid="line-chart"]').should('be.visible');
    
    // Check chart titles
    cy.contains('Vulnerability Types').should('be.visible');
    cy.contains('Severity Distribution').should('be.visible');
    cy.contains('Risk Score Trend').should('be.visible');
  });

  it('should show recent reports list', () => {
    cy.contains('Dashboard').click();
    
    // Check recent reports section
    cy.contains('Recent Reports').should('be.visible');
    
    // Should show report entries
    cy.get('[data-testid="report-item"]').should('have.length.at.least', 1);
  });

  it('should handle report selection', () => {
    cy.contains('Dashboard').click();
    
    // Click on a report
    cy.get('[data-testid="report-item"]').first().click();
    
    // Should navigate to report view
    cy.contains('Vulnerability Report').should('be.visible');
    cy.contains('Back to Dashboard').should('be.visible');
  });

  it('should refresh dashboard data', () => {
    cy.contains('Dashboard').click();
    
    // Click refresh button
    cy.get('[data-testid="refresh-button"]').click();
    
    // Should show refresh feedback
    cy.contains('Dashboard refreshed').should('be.visible');
  });

  it('should display system status', () => {
    cy.contains('Dashboard').click();
    
    // Check system status indicators
    cy.contains('System Status').should('be.visible');
    cy.contains('Models Loaded').should('be.visible');
    cy.contains('Active Requests').should('be.visible');
  });

  it('should show vulnerability statistics', () => {
    cy.contains('Dashboard').click();
    
    // Check vulnerability stats
    cy.contains('Vulnerability Types').should('be.visible');
    cy.contains('reentrancy').should('be.visible');
    cy.contains('integer_overflow').should('be.visible');
  });

  it('should handle empty state', () => {
    // Mock empty dashboard
    cy.intercept('GET', '/api/v1/metrics', {
      statusCode: 200,
      body: {
        total_requests: 0,
        successful_requests: 0,
        failed_requests: 0,
        average_processing_time: 0,
        model_usage_stats: {},
        vulnerability_type_stats: {},
        time_range: {
          start: new Date().toISOString(),
          end: new Date().toISOString(),
        },
      },
    }).as('emptyMetrics');
    
    cy.contains('Dashboard').click();
    
    // Should show empty state
    cy.contains('No analyses yet').should('be.visible');
  });
});
