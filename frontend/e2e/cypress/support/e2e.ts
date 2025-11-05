// Import commands.js using ES2015 syntax:
import './commands';

// Alternatively you can use CommonJS syntax:
// require('./commands')

// Hide fetch/XHR requests from command log
Cypress.on('window:before:load', (win) => {
  // Remove fetch from window to avoid logging
  if ('fetch' in win) {
    delete (win as any).fetch;
  }
});

// Handle uncaught exceptions
Cypress.on('uncaught:exception', (err, runnable) => {
  // Don't fail the test on uncaught exceptions
  return false;
});

// Custom commands for testing
declare global {
  namespace Cypress {
    interface Chainable {
      /**
       * Custom command to select DOM element by data-cy attribute.
       * @example cy.dataCy('greeting')
       */
      dataCy(value: string): Chainable<Element>;
      
      /**
       * Custom command to upload a file
       * @example cy.uploadFile('input[type="file"]', 'test.sol')
       */
      uploadFile(selector: string, fileName: string, fileContent?: string): Chainable<Element>;
      
      /**
       * Custom command to wait for API response
       * @example cy.waitForApi('@analyzeContract')
       */
      waitForApi(alias: string): Chainable<Element>;
      
      /**
       * Custom command to check if element is visible and clickable
       * @example cy.checkClickable('[data-testid="button"]')
       */
      checkClickable(selector: string): Chainable<Element>;
    }
  }
}

// Custom command implementations
Cypress.Commands.add('dataCy', (value: string) => {
  return cy.get(`[data-cy=${value}]`) as unknown as Cypress.Chainable<Element>;
});

Cypress.Commands.add('uploadFile', (selector: string, fileName: string, fileContent: string = 'contract Test { function test() public {} }') => {
  const file = new File([fileContent], fileName, { type: 'text/plain' });
  
  return cy.get(selector).then(input => {
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    const nativeInput = input[0] as HTMLInputElement;
    nativeInput.files = dataTransfer.files;
    
    return cy.wrap(input).trigger('change', { force: true });
  }) as unknown as Cypress.Chainable<Element>;
});

Cypress.Commands.add('waitForApi', (alias: string) => {
  return cy.wait(alias).then((interception) => {
    expect(interception.response?.statusCode).to.be.oneOf([200, 201, 202]);
    return cy.wrap(interception);
  }) as unknown as Cypress.Chainable<Element>;
});

Cypress.Commands.add('checkClickable', (selector: string) => {
  return cy.get(selector)
    .should('be.visible')
    .should('not.be.disabled')
    .should('not.have.attr', 'disabled') as unknown as Cypress.Chainable<Element>;
});

// Mock API responses for testing
Cypress.Commands.add('mockApi', () => {
  // Mock health check
  cy.intercept('GET', '/api/v1/health', {
    statusCode: 200,
    body: {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      version: '1.0.0',
      models_loaded: {
        codebert: true,
        gnn: true,
      },
      uptime: 3600,
      memory_usage: {
        total: 8589934592,
        available: 4294967296,
        percent: 50,
        used: 4294967296,
      },
    },
  }).as('healthCheck');

  // Mock contract analysis
  cy.intercept('POST', '/api/v1/analyze', {
    statusCode: 200,
    body: {
      success: true,
      message: 'Analysis completed successfully',
      report_id: 'test-report-id-123',
      estimated_processing_time: 2.5,
    },
  }).as('analyzeContract');

  // Mock report retrieval
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
        is_vulnerable: true,
        overall_confidence: 0.85,
        risk_score: 7.5,
        vulnerabilities: [
          {
            type: 'reentrancy',
            severity: 'high',
            confidence: 0.9,
            description: 'Reentrancy vulnerability detected',
            location: 'Line 5: msg.sender.transfer(amount);',
            explanation: 'External call before state change',
            recommendation: 'Use checks-effects-interactions pattern',
            references: ['https://example.com/reentrancy'],
          },
        ],
        optimization_suggestions: [
          {
            type: 'storage_optimization',
            description: 'Consider using smaller data types',
            potential_savings: '10-30%',
            implementation: 'Use uint128 instead of uint256 where possible',
            priority: 'medium',
          },
        ],
        model_predictions: [
          {
            model_type: 'codebert',
            is_vulnerable: true,
            confidence: 0.85,
            vulnerability_types: ['reentrancy'],
            processing_time: 1.2,
          },
          {
            model_type: 'gnn',
            is_vulnerable: true,
            confidence: 0.8,
            vulnerability_types: ['reentrancy'],
            processing_time: 0.8,
          },
        ],
        processing_time: 2.5,
        model_versions: {
          codebert: '1.0.0',
          gnn: '1.0.0',
        },
      },
    },
  }).as('getReport');

  // Mock system status
  cy.intercept('GET', '/api/v1/status', {
    statusCode: 200,
    body: {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      models: [
        {
          model_type: 'codebert',
          version: '1.0.0',
          is_loaded: true,
          load_time: 2.5,
          last_used: Date.now() - 1000,
          performance_metrics: {
            accuracy: 0.92,
            precision: 0.89,
            recall: 0.91,
          },
        },
        {
          model_type: 'gnn',
          version: '1.0.0',
          is_loaded: true,
          load_time: 3.2,
          last_used: Date.now() - 2000,
          performance_metrics: {
            accuracy: 0.88,
            precision: 0.85,
            recall: 0.87,
          },
        },
      ],
      system_metrics: {
        cpu_percent: 45.2,
        memory: {
          total: 8589934592,
          available: 4294967296,
          used: 4294967296,
          percent: 50,
        },
        disk: {
          total: 107374182400,
          used: 53687091200,
          free: 53687091200,
          percent: 50,
        },
      },
      active_requests: 2,
      queue_size: 0,
    },
  }).as('getSystemStatus');

  // Mock metrics
  cy.intercept('GET', '/api/v1/metrics', {
    statusCode: 200,
    body: {
      total_requests: 150,
      successful_requests: 142,
      failed_requests: 8,
      average_processing_time: 2.3,
      model_usage_stats: {
        codebert: 75,
        gnn: 60,
        ensemble: 15,
      },
      vulnerability_type_stats: {
        reentrancy: 25,
        integer_overflow: 18,
        access_control: 12,
        unchecked_external_calls: 8,
      },
      time_range: {
        start: new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(),
        end: new Date().toISOString(),
      },
    },
  }).as('getMetrics');
});

// Add the mockApi command to the global namespace
declare global {
  namespace Cypress {
    interface Chainable {
      mockApi(): Chainable<Element>;
    }
  }
}
