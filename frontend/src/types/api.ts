// API Types for Mock Data Generator - Backend Integration

export interface APIResponse<T = unknown> {
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

export interface GenerationRequest {
  input_data: Record<string, unknown>[];
  count: number;
  enable_moderation?: boolean;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  cache_expiration?: boolean;
  user_id?: string;
  session_id?: string;
}

export interface GenerateMockDataResponse extends APIResponse {
  data: Record<string, unknown>[];
}

export interface GenerationParams {
  count: number;
}

// Legacy types for backward compatibility
export interface ChunkingStats {
  total_chunks: number;
  successful_chunks: number;
  failed_chunks: number;
  total_processing_time: number;
  average_chunk_time: number;
  parallel_efficiency: number;
  retry_attempts: number;
}

export interface GenerationMetadata {
  request_id: string;
  processing_method: string;
  chunked: boolean;
  user_id?: string;
  session_id?: string;
  generated_count: number;
  requested_count: number;
  use_case: string;
  total_processing_time: number;
  input_validation: string;
  content_moderation: string;
  is_large_request: boolean;
  generation_method?: string;
  was_chunked: boolean;
  controller: string;
  client_ip: string;
  response_time_seconds: number;
  timestamp: number;
  chunking_stats?: ChunkingStats;
}

export interface ModerationResult {
  status: string;
  items_checked?: number;
  moderation_score?: number;
  flagged_categories?: string[];
  reason?: string;
}

export interface GenerationResponse {
  mock_data: Record<string, unknown>[];
  metadata: GenerationMetadata;
  request_id: string;
  success: boolean;
  moderation_results?: ModerationResult;
}

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy';
  timestamp: number;
  controller_stats: Record<string, unknown>;
  use_case_health: Record<string, unknown>;
  capabilities: Record<string, unknown>;
}

export interface SystemDiagnostics {
  controller_diagnostics: Record<string, unknown>;
  use_case_diagnostics: Record<string, unknown>;
  timestamp: number;
}

export interface RealTimeMetrics {
  request_rate: number;
  success_rate: number;
  average_response_time: number;
  queue_utilization: number;
  active_sessions: number;
  cache_hit_rate: number;
}

export interface SavedResultsSummary {
  files: Array<{
    id: string;
    name: string;
    size: number;
    created_at: string;
    record_count: number;
  }>;
  statistics: {
    total_files: number;
    total_records: number;
    total_size: number;
  };
  metrics: {
    most_used_template: string;
    average_generation_time: number;
    popular_record_counts: number[];
  };
}

export interface Template {
  id: string;
  name: string;
  description: string;
  category: string;
  example_data: Record<string, unknown>[];
  tags: string[];
  created_at: string;
  usage_count: number;
}

export interface ApiError {
  message: string;
  code: string;
  details?: unknown;
  timestamp: string;
}