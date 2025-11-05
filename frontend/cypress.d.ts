/// <reference types="cypress" />

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
      
      /**
       * Custom command to mock API responses
       * @example cy.mockApi()
       */
      mockApi(): Chainable<Element>;
    }
  }
}

export {};
