import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  Activity, 
  Server, 
  Database, 
  Zap, 
  Shield, 
  Clock, 
  TrendingUp,
  AlertCircle,
  CheckCircle,
  XCircle,
  Cpu,
  MemoryStick,
  HardDrive
} from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { MockDataGeneratorAPI } from '@/services/api';
import type { HealthStatus, SystemDiagnostics, RealTimeMetrics } from '@/types/api';

interface SystemMonitorProps {
  className?: string;
}

export const SystemMonitor: React.FC<SystemMonitorProps> = ({ className }) => {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [diagnostics, setDiagnostics] = useState<SystemDiagnostics | null>(null);
  const [metrics, setMetrics] = useState<RealTimeMetrics | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());

  const fetchSystemData = async () => {
    try {
      const [health, diag, metr] = await Promise.all([
        MockDataGeneratorAPI.getSystemHealth(),
        MockDataGeneratorAPI.getSystemDiagnostics(),
        MockDataGeneratorAPI.getRealTimeMetrics(),
      ]);
      
      setHealthStatus(health);
      setDiagnostics(diag);
      setMetrics(metr);
      setLastUpdate(new Date());
      setIsLoading(false);
    } catch (error) {
      console.error('Failed to fetch system data:', error);
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchSystemData();
    const interval = setInterval(fetchSystemData, 30000); // Update every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-brand-accent bg-brand-accent/20 border-brand-accent/30';
      case 'degraded': return 'text-yellow-500 bg-yellow-500/20 border-yellow-500/30';
      case 'unhealthy': return 'text-red-500 bg-red-500/20 border-red-500/30';
      default: return 'text-muted-foreground bg-muted border-muted-foreground/30';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="w-4 h-4" />;
      case 'degraded': return <AlertCircle className="w-4 h-4" />;
      case 'unhealthy': return <XCircle className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className={className}
      >
        <Card className="bg-gradient-to-br from-background via-background to-brand-glass/20 border-brand-primary/20">
          <CardContent className="p-8">
            <div className="flex items-center justify-center space-y-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="w-8 h-8 border-2 border-brand-primary/20 border-t-brand-primary rounded-full"
              />
              <span className="ml-3 text-muted-foreground">Loading system status...</span>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={className}
    >
      <Card className="bg-gradient-to-br from-background via-background to-brand-glass/20 border-brand-primary/20 shadow-elegant">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-lg font-semibold bg-gradient-primary bg-clip-text text-transparent">
              <Activity className="w-5 h-5 text-brand-primary" />
              System Monitor
            </CardTitle>
            
            <div className="flex items-center gap-2">
              {healthStatus && (
                <Badge className={getStatusColor(healthStatus.status)}>
                  {getStatusIcon(healthStatus.status)}
                  <span className="ml-1 capitalize">{healthStatus.status}</span>
                </Badge>
              )}
              <div className="text-xs text-muted-foreground">
                Updated {lastUpdate.toLocaleTimeString()}
              </div>
            </div>
          </div>
        </CardHeader>

        <CardContent>
          <Tabs defaultValue="overview" className="space-y-4">
            <TabsList className="grid w-full grid-cols-4 bg-brand-glass/20 border border-brand-primary/20">
              <TabsTrigger 
                value="overview"
                className="data-[state=active]:bg-brand-primary data-[state=active]:text-white"
              >
                Overview
              </TabsTrigger>
              <TabsTrigger 
                value="performance"
                className="data-[state=active]:bg-brand-primary data-[state=active]:text-white"
              >
                Performance
              </TabsTrigger>
              <TabsTrigger 
                value="metrics"
                className="data-[state=active]:bg-brand-primary data-[state=active]:text-white"
              >
                Metrics
              </TabsTrigger>
              <TabsTrigger 
                value="diagnostics"
                className="data-[state=active]:bg-brand-primary data-[state=active]:text-white"
              >
                Diagnostics
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-4">
              {healthStatus && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="flex items-center justify-between mb-2">
                      <Server className="w-5 h-5 text-brand-primary" />
                      <Badge className="bg-brand-accent/20 text-brand-accent">
                        {healthStatus.controller_stats.active_workers}
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">Active Workers</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="flex items-center justify-between mb-2">
                      <Zap className="w-5 h-5 text-brand-secondary" />
                      <Badge className="bg-brand-secondary/20 text-brand-secondary">
                        {healthStatus.controller_stats.total_requests.toLocaleString()}
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">Total Requests</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="flex items-center justify-between mb-2">
                      <Clock className="w-5 h-5 text-brand-accent" />
                      <Badge className="bg-brand-accent/20 text-brand-accent">
                        {healthStatus.controller_stats.average_response_time.toFixed(0)}ms
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">Avg Response</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="flex items-center justify-between mb-2">
                      <TrendingUp className="w-5 h-5 text-brand-primary" />
                      <Badge className="bg-brand-primary/20 text-brand-primary">
                        {Math.floor(healthStatus.uptime / 3600)}h
                      </Badge>
                    </div>
                    <div className="text-sm text-muted-foreground">Uptime</div>
                  </div>
                </div>
              )}

              {healthStatus?.capabilities && (
                <div className="space-y-3">
                  <h4 className="font-semibold text-brand-primary">System Capabilities</h4>
                  <div className="flex flex-wrap gap-2">
                    {healthStatus.capabilities.map((capability, index) => (
                      <Badge 
                        key={index}
                        variant="secondary"
                        className="bg-brand-primary/10 text-brand-primary border border-brand-primary/20"
                      >
                        {capability}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="performance" className="space-y-4">
              {diagnostics && (
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                      <div className="flex items-center gap-2 mb-3">
                        <Cpu className="w-5 h-5 text-brand-primary" />
                        <span className="font-semibold">CPU Usage</span>
                      </div>
                      <Progress 
                        value={diagnostics.controller_diagnostics.cpu_usage * 100} 
                        className="mb-2"
                      />
                      <div className="text-sm text-muted-foreground">
                        {(diagnostics.controller_diagnostics.cpu_usage * 100).toFixed(1)}%
                      </div>
                    </div>

                    <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                      <div className="flex items-center gap-2 mb-3">
                        <MemoryStick className="w-5 h-5 text-brand-secondary" />
                        <span className="font-semibold">Memory Usage</span>
                      </div>
                      <Progress 
                        value={diagnostics.controller_diagnostics.memory_usage * 100} 
                        className="mb-2"
                      />
                      <div className="text-sm text-muted-foreground">
                        {(diagnostics.controller_diagnostics.memory_usage * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                      <div className="flex items-center gap-2 mb-3">
                        <HardDrive className="w-5 h-5 text-brand-accent" />
                        <span className="font-semibold">Queue Length</span>
                      </div>
                      <div className="text-2xl font-bold text-brand-accent">
                        {diagnostics.controller_diagnostics.queue_length}
                      </div>
                      <div className="text-sm text-muted-foreground">Pending requests</div>
                    </div>

                    <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                      <div className="flex items-center gap-2 mb-3">
                        <AlertCircle className="w-5 h-5 text-red-500" />
                        <span className="font-semibold">Error Rate</span>
                      </div>
                      <div className="text-2xl font-bold text-red-500">
                        {(diagnostics.controller_diagnostics.error_rate * 100).toFixed(2)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Last 24h</div>
                    </div>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="metrics" className="space-y-4">
              {metrics && (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="text-sm text-muted-foreground mb-1">Request Rate</div>
                    <div className="text-2xl font-bold text-brand-primary">
                      {metrics.request_rate.toFixed(1)}
                    </div>
                    <div className="text-xs text-muted-foreground">req/sec</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="text-sm text-muted-foreground mb-1">Success Rate</div>
                    <div className="text-2xl font-bold text-brand-accent">
                      {(metrics.success_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">last hour</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="text-sm text-muted-foreground mb-1">Avg Response Time</div>
                    <div className="text-2xl font-bold text-brand-secondary">
                      {metrics.average_response_time.toFixed(0)}ms
                    </div>
                    <div className="text-xs text-muted-foreground">last hour</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="text-sm text-muted-foreground mb-1">Queue Utilization</div>
                    <div className="text-2xl font-bold text-brand-primary">
                      {(metrics.queue_utilization * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">current</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="text-sm text-muted-foreground mb-1">Active Sessions</div>
                    <div className="text-2xl font-bold text-brand-accent">
                      {metrics.active_sessions}
                    </div>
                    <div className="text-xs text-muted-foreground">current</div>
                  </div>

                  <div className="bg-brand-glass/20 p-4 rounded-lg border border-brand-primary/20">
                    <div className="text-sm text-muted-foreground mb-1">Cache Hit Rate</div>
                    <div className="text-2xl font-bold text-brand-secondary">
                      {(metrics.cache_hit_rate * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">last hour</div>
                  </div>
                </div>
              )}
            </TabsContent>

            <TabsContent value="diagnostics" className="space-y-4">
              {diagnostics?.use_case_health && (
                <div className="space-y-3">
                  <h4 className="font-semibold text-brand-primary">Use Case Health</h4>
                  <div className="space-y-2">
                    {Object.entries(diagnostics.use_case_health).map(([useCase, health]) => (
                      <div 
                        key={useCase}
                        className="flex items-center justify-between p-3 bg-brand-glass/20 rounded-lg border border-brand-primary/20"
                      >
                        <div>
                          <div className="font-medium">{useCase}</div>
                          <div className="text-xs text-muted-foreground">
                            Last check: {new Date(health.last_check).toLocaleString()}
                          </div>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge className={getStatusColor(health.status)}>
                            {getStatusIcon(health.status)}
                            <span className="ml-1 capitalize">{health.status}</span>
                          </Badge>
                          <div className="text-sm font-medium">
                            {(health.success_rate * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </motion.div>
  );
};