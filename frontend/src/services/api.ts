import axios, { AxiosResponse } from 'axios';
import toast from 'react-hot-toast';
import { config, API_ENDPOINTS } from '@/config/config';
import type {
  GenerationRequest,
  GenerateMockDataResponse,
  HealthStatus,
  SystemDiagnostics,
  RealTimeMetrics,
  SavedResultsSummary,
  Template,
  ApiError
} from '@/types/api';

// API Configuration
const API_BASE_URL = config.API_BASE_URL;
const MODEL_SERVER_URL = config.MODEL_SERVER_URL;

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: config.API_TIMEOUT,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for adding auth tokens, request IDs, etc.
apiClient.interceptors.request.use(
  (config) => {
    // Add request ID for tracking
    config.headers['X-Request-ID'] = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Add user session if available
    const sessionId = localStorage.getItem('mdg_session_id');
    if (sessionId) {
      config.headers['X-Session-ID'] = sessionId;
    }
    
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    const apiError: ApiError = {
      message: error.response?.data?.message || error.message || 'An unexpected error occurred',
      code: error.response?.data?.code || error.code || 'UNKNOWN_ERROR',
      details: error.response?.data?.details,
      timestamp: new Date().toISOString(),
    };

    // Show user-friendly error messages
    if (error.response?.status === 429) {
      toast.error('Rate limit exceeded. Please wait a moment before trying again.');
    } else if (error.response?.status >= 500) {
      toast.error('Server error. Our team has been notified.');
    } else if (error.response?.status === 400) {
      toast.error(apiError.message);
    } else if (!navigator.onLine) {
      toast.error('No internet connection. Please check your network.');
    } else {
      toast.error(apiError.message);
    }

    return Promise.reject(apiError);
  }
);

// API Service Class
export class MockDataGeneratorAPI {
  
  // Primary data generation - Using the same endpoint as working frontend
  static async generateMockData(
    examples: Record<string, unknown>[], 
    count: number,
    options?: {
      enable_moderation?: boolean;
      temperature?: number;
      max_tokens?: number;
      top_p?: number;
      cache_expiration?: boolean;
    }
  ): Promise<GenerateMockDataResponse> {
    const params: Record<string, string | number | boolean> = { count };
    
    // Add optional parameters if provided
    if (options?.enable_moderation !== undefined) {
      params.enable_moderation = options.enable_moderation;
    }
    if (options?.temperature !== undefined) {
      params.temperature = options.temperature;
    }
    if (options?.max_tokens !== undefined) {
      params.max_tokens = options.max_tokens;
    }
    if (options?.top_p !== undefined) {
      params.top_p = options.top_p;
    }
    if (options?.cache_expiration !== undefined) {
      params.cache_expiration = options.cache_expiration;
    }

    const response: AxiosResponse<GenerateMockDataResponse> = await apiClient.post(
      '/generate-mock-data',
      examples,
      { params }
    );
    return response.data;
  }

  // Legacy endpoint for backward compatibility
  static async generateLegacy(examples: Record<string, unknown>[], count: number): Promise<GenerateMockDataResponse> {
    const response: AxiosResponse<GenerateMockDataResponse> = await apiClient.post(
      API_ENDPOINTS.LEGACY_GENERATE,
      { examples, count }
    );
    return response.data;
  }

  // Direct backend integration - using the main generation endpoint
  static async generateWithBackend(
    examples: Record<string, unknown>[], 
    count: number, 
    options?: {
      enable_moderation?: boolean;
      temperature?: number;
      max_tokens?: number;
      top_p?: number;
      cache_expiration?: boolean;
    }
  ): Promise<GenerateMockDataResponse> {
    // Use the same simple format as working frontend
    return this.generateMockData(examples, count, options);
  }

  // System health monitoring
  static async getSystemHealth(): Promise<HealthStatus> {
    const response: AxiosResponse<HealthStatus> = await apiClient.get('/health');
    return response.data;
  }

  // System diagnostics
  static async getSystemDiagnostics(): Promise<SystemDiagnostics> {
    const response: AxiosResponse<SystemDiagnostics> = await apiClient.get(API_ENDPOINTS.DIAGNOSTICS);
    return response.data;
  }

  // Real-time metrics
  static async getRealTimeMetrics(): Promise<RealTimeMetrics> {
    const response: AxiosResponse<RealTimeMetrics> = await apiClient.get(API_ENDPOINTS.METRICS);
    return response.data;
  }

  // Saved results
  static async getSavedResults(): Promise<SavedResultsSummary> {
    const response: AxiosResponse<SavedResultsSummary> = await apiClient.get(API_ENDPOINTS.SAVED_RESULTS);
    return response.data;
  }

  // Monitoring endpoints
  static async getMonitoringHealth(): Promise<HealthStatus> {
    const response: AxiosResponse<HealthStatus> = await apiClient.get(API_ENDPOINTS.MONITORING_HEALTH);
    return response.data;
  }

  static async getLLMStats(): Promise<Record<string, unknown>> {
    const response = await apiClient.get(API_ENDPOINTS.LLM_STATS);
    return response.data;
  }

  static async getModelServerStats(): Promise<Record<string, unknown>> {
    const response = await apiClient.get(API_ENDPOINTS.MODEL_SERVER_STATS);
    return response.data;
  }

  static async getCombinedStats(): Promise<Record<string, unknown>> {
    const response = await apiClient.get(API_ENDPOINTS.COMBINED_STATS);
    return response.data;
  }

  // Template management (for future implementation)
  static async getTemplates(): Promise<Template[]> {
    try {
      const response: AxiosResponse<Template[]> = await apiClient.get('/templates');
      return response.data;
    } catch (error) {
      // Return default templates if endpoint doesn't exist yet
      return this.getDefaultTemplates();
    }
  }

  static async saveTemplate(template: Omit<Template, 'id' | 'created_at' | 'usage_count'>): Promise<Template> {
    const response: AxiosResponse<Template> = await apiClient.post('/templates', template);
    return response.data;
  }

  // Utility methods
  static async testConnection(): Promise<boolean> {
    try {
      await this.getSystemHealth();
      return true;
    } catch (error) {
      return false;
    }
  }

  static getDefaultTemplates(): Template[] {
    return [
      {
        id: 'user-profile',
        name: 'User Profile',
        description: 'Generate realistic user profile data',
        category: 'Users',
        example_data: [
          {
            id: 1,
            firstName: 'John',
            lastName: 'Doe',
            email: 'john.doe@example.com',
            age: 28,
            profession: 'Software Engineer',
            location: 'San Francisco, CA',
            salary: 120000,
            skills: ['JavaScript', 'React', 'Node.js'],
            isActive: true
          }
        ],
        tags: ['user', 'profile', 'personal'],
        created_at: new Date().toISOString(),
        usage_count: 0
      },
      {
        id: 'product-catalog',
        name: 'Product Catalog',
        description: 'Generate e-commerce product data',
        category: 'E-commerce',
        example_data: [
          {
            id: 'PROD-001',
            name: 'Wireless Bluetooth Headphones',
            category: 'Electronics',
            price: 89.99,
            brand: 'TechAudio',
            description: 'High-quality wireless headphones with noise cancellation',
            inStock: true,
            rating: 4.5,
            reviews: 127
          }
        ],
        tags: ['product', 'ecommerce', 'catalog'],
        created_at: new Date().toISOString(),
        usage_count: 0
      },
      {
        id: 'transaction-data',
        name: 'Transaction Data',
        description: 'Generate financial transaction records',
        category: 'Finance',
        example_data: [
          {
            transactionId: 'TXN-2024-001',
            amount: 156.78,
            currency: 'USD',
            merchant: 'Amazon.com',
            category: 'Shopping',
            timestamp: '2024-01-15T10:30:00Z',
            status: 'completed',
            paymentMethod: 'credit_card'
          }
        ],
        tags: ['finance', 'transaction', 'payment'],
        created_at: new Date().toISOString(),
        usage_count: 0
      }
    ];
  }
}

// Export default instance for backward compatibility
export default MockDataGeneratorAPI;