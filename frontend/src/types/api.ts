export interface APIResponse<T = any> {
  code: string;
  message: string;
  usedFromCache: boolean;
  data: T;
  error: string;
}

export interface GenerateMockDataRequest {
  examples: Record<string, any>[];
}

export interface GenerateMockDataResponse extends APIResponse {
  data: Record<string, any>[];
}

export interface GenerationParams {
  count: number;
} 