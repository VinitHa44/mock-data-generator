// Frontend Configuration
export const config = {
  // API Configuration
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api',
  MODEL_SERVER_URL: import.meta.env.VITE_MODEL_SERVER_URL || 'http://localhost:8001',
  
  // App Configuration
  APP_NAME: import.meta.env.VITE_APP_NAME || 'Mock Data Generator',
  APP_VERSION: import.meta.env.VITE_APP_VERSION || '4.0.0',
  
  // Development Settings
  DEV_MODE: import.meta.env.VITE_DEV_MODE === 'true',
  ENABLE_DEBUG: import.meta.env.VITE_ENABLE_DEBUG === 'true',
  
  // API Timeouts
  // API_TIMEOUT: 30000, // 30 seconds
  // GENERATION_TIMEOUT: 300000, // 5 minutes for long generations
  
  // Feature Flags
  ENABLE_MODERATION: true,
  ENABLE_CACHING: true,
  ENABLE_ANALYTICS: true,
  
  // UI Configuration
  MAX_GENERATION_COUNT: 500,
  DEFAULT_GENERATION_COUNT: 10,
  MAX_EXAMPLE_COUNT: 50,
  
  // Cache Configuration
  CACHE_DURATION: 5 * 60 * 1000, // 5 minutes
  SESSION_DURATION: 24 * 60 * 60 * 1000, // 24 hours
};

// Environment-specific configurations
export const isDevelopment = config.DEV_MODE;
export const isProduction = !config.DEV_MODE;

// API endpoints
export const API_ENDPOINTS = {
  // Generation endpoints
  GENERATE: '/generate-mock-data',
  LEGACY_GENERATE: '/generation/legacy',
  HEALTH: '/health',
  DIAGNOSTICS: '/generation/diagnostics',
  METRICS: '/generation/metrics',
  SAVED_RESULTS: '/generation/saved-results',
  
  // Monitoring endpoints
  MONITORING_HEALTH: '/monitoring/health',
  LLM_STATS: '/monitoring/stats/llm',
  MODEL_SERVER_STATS: '/monitoring/stats/model-server',
  COMBINED_STATS: '/monitoring/stats/combined',
  
  // Template endpoints (future)
  TEMPLATES: '/templates',
} as const; 