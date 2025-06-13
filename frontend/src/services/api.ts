import axios from 'axios';
import { GenerateMockDataResponse, GenerationParams } from '../types/api';
import { config } from '../config/config';

const API_BASE_URL = config.API_BASE_URL;

const api = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes timeout for LLM processing
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const mockDataApi = {
  generateMockData: async (
    examples: Record<string, any>[], 
    params: GenerationParams
  ): Promise<GenerateMockDataResponse> => {
    const response = await api.post('/generate-mock-data', examples, {
      params: {
        count: params.count,
      },
    });
    return response.data;
  },

  healthCheck: async (): Promise<{ status: string }> => {
    const response = await api.get('/health');
    return response.data;
  },
};

export default api; 