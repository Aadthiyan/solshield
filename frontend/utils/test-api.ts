// Test API connection
import { healthCheck, checkConnection } from './api';

export const testApiConnection = async () => {
  try {
    console.log('Testing API connection...');
    
    // Test basic connection
    const isConnected = await checkConnection();
    console.log('Connection status:', isConnected);
    
    if (isConnected) {
      // Test health endpoint
      const healthResponse = await healthCheck();
      console.log('Health check response:', healthResponse);
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('API connection test failed:', error);
    return false;
  }
};

// Test API endpoints
export const testApiEndpoints = async () => {
  const results = {
    health: false,
    status: false,
    metrics: false,
  };

  try {
    // Test health endpoint
    const healthResponse = await healthCheck();
    results.health = healthResponse.status === 200;
    console.log('Health endpoint:', results.health ? 'OK' : 'FAILED');
  } catch (error) {
    console.error('Health endpoint test failed:', error);
  }

  return results;
};
