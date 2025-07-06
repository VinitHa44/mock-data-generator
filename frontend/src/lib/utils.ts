import { clsx, type ClassValue } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Error handling utilities
export interface ErrorInfo {
  isNetworkError: boolean;
  isConnectionError: boolean;
  isTimeoutError: boolean;
  isServerError: boolean;
  isClientError: boolean;
  userFriendlyMessage: string;
  shouldRetry: boolean;
}

// Type for extended error objects
interface ExtendedError extends Error {
  code?: string;
  response?: {
    status?: number;
    data?: {
      message?: string;
    };
  };
}

export function classifyError(error: unknown): ErrorInfo {
  const errorMessage = error instanceof Error ? error.message : String(error);
  const extendedError = error as ExtendedError;
  const errorCode = extendedError?.code || '';
  
  const isNetworkError = (errorCode === 'ERR_NETWORK' && !extendedError?.response) || 
                        errorCode === 'ERR_CONNECTION_REFUSED' ||
                        errorCode === 'ERR_NAME_NOT_RESOLVED';
  
  const isConnectionError = errorCode === 'ECONNREFUSED' || 
                           errorCode === 'ERR_CONNECTION_REFUSED' ||
                           (errorMessage.includes('connection refused') && !extendedError?.response);
  
  const isTimeoutError = errorCode === 'ETIMEDOUT' || 
                        errorCode === 'ERR_TIMEOUT' ||
                        errorMessage.includes('timeout');
  
  const isServerError = extendedError?.response?.status ? extendedError.response.status >= 500 : false;
  const isClientError = extendedError?.response?.status ? extendedError.response.status >= 400 && extendedError.response.status < 500 : false;
  
  let userFriendlyMessage = 'An unexpected error occurred. Please try again.';
  let shouldRetry = false;
  
  if (isConnectionError) {
    userFriendlyMessage = 'Cannot connect to the server. Please ensure the backend service is running on the correct port.';
    shouldRetry = true;
  } else if (isTimeoutError) {
    userFriendlyMessage = 'Request timed out. The server is taking too long to respond. Please try again.';
    shouldRetry = true;
  } else if (isNetworkError && !extendedError?.response) {
    // Only show network error if there's no response (indicating connection refused)
    userFriendlyMessage = 'Cannot connect to the server. Please ensure the backend service is running.';
    shouldRetry = true;
  } else if (isServerError) {
    userFriendlyMessage = 'Server error. Our team has been notified. Please try again later.';
    shouldRetry = true;
  } else if (isClientError) {
    const status = extendedError?.response?.status;
    if (status === 404) {
      userFriendlyMessage = 'Service endpoint not found. Please check if the backend is running.';
    } else if (status === 403) {
      userFriendlyMessage = 'Access denied. Please check your permissions.';
    } else if (status === 401) {
      userFriendlyMessage = 'Authentication required. Please log in again.';
    } else if (status === 429) {
      userFriendlyMessage = 'Rate limit exceeded. Please wait a moment before trying again.';
      shouldRetry = true;
    } else {
      userFriendlyMessage = extendedError?.response?.data?.message || 'Request failed. Please check your input and try again.';
    }
  }
  
  return {
    isNetworkError,
    isConnectionError,
    isTimeoutError,
    isServerError,
    isClientError,
    userFriendlyMessage,
    shouldRetry
  };
}

export function getErrorMessage(error: unknown): string {
  return classifyError(error).userFriendlyMessage;
}

export function shouldRetryError(error: unknown): boolean {
  return classifyError(error).shouldRetry;
}

// Test function for error handling (development only)
export function simulateError(errorType: 'network' | 'connection' | 'timeout' | 'server' | 'client'): Error {
  switch (errorType) {
    case 'network': {
      return new Error('Network Error');
    }
    case 'connection': {
      const connError = new Error('connect ECONNREFUSED 127.0.0.1:8000') as ExtendedError;
      connError.code = 'ECONNREFUSED';
      return connError;
    }
    case 'timeout': {
      const timeoutError = new Error('timeout of 30000ms exceeded') as ExtendedError;
      timeoutError.code = 'ETIMEDOUT';
      return timeoutError;
    }
    case 'server': {
      const serverError = new Error('Internal Server Error') as ExtendedError;
      serverError.response = { status: 500 };
      return serverError;
    }
    case 'client': {
      const clientError = new Error('Bad Request') as ExtendedError;
      clientError.response = { status: 400 };
      return clientError;
    }
    default: {
      return new Error('Unknown error');
    }
  }
}
