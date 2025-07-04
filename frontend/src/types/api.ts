export interface APIResponse<T = any> {
  code: string;
  message: string;
  usedFromCache: boolean;
  data: T;
  error: string;
  cacheInfo?: {
    cachedCount: number;
    generatedCount: number;
    totalCount: number;
    cacheHitType: 'full' | 'partial' | 'none';
  };
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