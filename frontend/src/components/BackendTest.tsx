import React, { useState } from 'react';
import { MockDataGeneratorAPI } from '@/services/api';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { toast } from 'react-hot-toast';
import type { HealthStatus, GenerateMockDataResponse } from '@/types/api';

export function BackendTest() {
  const [isLoading, setIsLoading] = useState(false);
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [generationResult, setGenerationResult] = useState<GenerateMockDataResponse | null>(null);

  const testHealth = async () => {
    setIsLoading(true);
    try {
      const health = await MockDataGeneratorAPI.getSystemHealth();
      setHealthStatus(health);
      toast.success('Backend health check successful!');
    } catch (error) {
      toast.error('Backend health check failed');
      console.error('Health check error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const testGeneration = async () => {
    setIsLoading(true);
    try {
      const examples = [
        {
          id: 1,
          name: "John Doe",
          email: "john@example.com",
          age: 30,
          profession: "Software Engineer"
        }
      ];

      const result = await MockDataGeneratorAPI.generateWithBackend(examples, 3, {
        enable_moderation: true,
        temperature: 0.7
      });

      setGenerationResult(result);
      toast.success(`Generated ${result.data.length} records successfully!`);
    } catch (error) {
      toast.error('Generation failed');
      console.error('Generation error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>Backend Connection Test</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex gap-2">
            <Button 
              onClick={testHealth} 
              disabled={isLoading}
              variant="outline"
            >
              Test Health
            </Button>
            <Button 
              onClick={testGeneration} 
              disabled={isLoading}
            >
              Test Generation
            </Button>
          </div>

          {healthStatus && (
            <div className="space-y-2">
              <h4 className="font-semibold">Health Status:</h4>
              <div className="flex gap-2">
                <Badge variant={healthStatus.status === 'healthy' ? 'default' : 'destructive'}>
                  {healthStatus.status}
                </Badge>
                <span className="text-sm text-muted-foreground">
                  Timestamp: {new Date(healthStatus.timestamp * 1000).toLocaleString()}
                </span>
              </div>
              <pre className="text-xs bg-muted p-2 rounded overflow-auto">
                {JSON.stringify(healthStatus, null, 2)}
              </pre>
            </div>
          )}

          {generationResult && (
            <div className="space-y-2">
              <h4 className="font-semibold">Generation Result:</h4>
              <div className="flex gap-2">
                <Badge variant="outline">
                  {generationResult.usedFromCache ? 'Cached' : 'Generated'}
                </Badge>
                <Badge variant="outline">
                  {generationResult.data.length} items
                </Badge>
                {generationResult.cacheInfo && (
                  <Badge variant="outline">
                    {generationResult.cacheInfo.cacheHitType}
                  </Badge>
                )}
              </div>
              <pre className="text-xs bg-muted p-2 rounded overflow-auto max-h-40">
                {JSON.stringify(generationResult.data, null, 2)}
              </pre>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
} 