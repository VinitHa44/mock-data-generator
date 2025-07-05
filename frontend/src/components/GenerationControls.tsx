import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Sliders, Shield, Zap, Settings, Sparkles, Play, Pause } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import type { GenerationRequest } from '@/types/api';

interface GenerationControlsProps {
  onGenerate: (params: Omit<GenerationRequest, 'input_data'>, sampleData?: string) => void;
  isGenerating: boolean;
  progress?: number;
}

export const GenerationControls: React.FC<GenerationControlsProps> = ({
  onGenerate,
  isGenerating,
  progress
}) => {
  const [count, setCount] = useState(50);
  const [enableModeration, setEnableModeration] = useState(true);
  const [temperature, setTemperature] = useState([0.7]);
  const [maxTokens, setMaxTokens] = useState([2048]);
  const [topP, setTopP] = useState([0.9]);
  const [cacheExpiration, setCacheExpiration] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleCountChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseInt(e.target.value);
    if (!isNaN(value) && value >= 1 && value <= 500) {
      setCount(value);
    }
  };

  const handleGenerate = useCallback(() => {
    const params: Omit<GenerationRequest, 'input_data'> = {
      count: count,
      enable_moderation: enableModeration,
      temperature: temperature[0],
      max_tokens: maxTokens[0],
      top_p: topP[0],
      cache_expiration: cacheExpiration,
      user_id: localStorage.getItem('mdg_user_id') || undefined,
      session_id: localStorage.getItem('mdg_session_id') || undefined,
    };
    
    onGenerate(params);
  }, [count, enableModeration, temperature, maxTokens, topP, cacheExpiration, onGenerate]);

  const resetToDefaults = () => {
    setCount(50);
    setTemperature([0.7]);
    setMaxTokens([2048]);
    setTopP([0.9]);
    setEnableModeration(true);
    setCacheExpiration(false);
  };

  const presets = [
    {
      name: 'Conservative',
      description: 'Safe, predictable results',
      settings: { count: 25, temperature: 0.3, maxTokens: 1024, topP: 0.8 }
    },
    {
      name: 'Balanced',
      description: 'Good mix of creativity and consistency',
      settings: { count: 50, temperature: 0.7, maxTokens: 2048, topP: 0.9 }
    },
    {
      name: 'Creative',
      description: 'More diverse and varied results',
      settings: { count: 100, temperature: 1.0, maxTokens: 4096, topP: 0.95 }
    }
  ];

  const applyPreset = (preset: typeof presets[0]) => {
    setCount(preset.settings.count);
    setTemperature([preset.settings.temperature]);
    setMaxTokens([preset.settings.maxTokens]);
    setTopP([preset.settings.topP]);
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.3, delay: 0.1 }}
    >
      <Card className="bg-gradient-to-br from-background via-background to-brand-glass/20 border-brand-secondary/20 shadow-elegant">
        <CardHeader className="pb-4">
          <CardTitle className="flex items-center gap-2 text-lg font-semibold bg-gradient-accent bg-clip-text text-transparent">
            <Sliders className="w-5 h-5 text-brand-accent" />
            Generation Controls
          </CardTitle>
        </CardHeader>

        <CardContent className="space-y-6">
          {/* Presets */}
          <div>
            <Label className="text-sm font-medium mb-3 block">Quick Presets</Label>
            <div className="grid grid-cols-3 gap-2">
              {presets.map((preset) => (
                <Button
                  key={preset.name}
                  variant="outline"
                  size="sm"
                  onClick={() => applyPreset(preset)}
                  disabled={isGenerating}
                  className="h-auto p-3 flex flex-col items-start border-brand-primary/30 hover:bg-brand-primary/10"
                >
                  <span className="font-medium text-xs">{preset.name}</span>
                  <span className="text-xs text-muted-foreground mt-1 leading-tight">
                    {preset.description}
                  </span>
                </Button>
              ))}
            </div>
          </div>

          <Separator className="bg-brand-primary/20" />

          {/* Record Count */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <Label className="text-sm font-medium">Records to Generate</Label>
              <Badge variant="secondary" className="bg-brand-primary/20 text-brand-primary">
                {count}
              </Badge>
            </div>
            <Input
              type="number"
              value={count}
              onChange={handleCountChange}
              min={1}
              max={500}
              disabled={isGenerating}
              className="text-center font-semibold text-lg border-brand-primary/30 focus:ring-brand-primary"
              placeholder="Enter count (1-500)"
            />
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>Min: 1</span>
              <span>Max: 500</span>
            </div>
            
            {/* Progress Bar for Generation */}
            {isGenerating && (
              <div className="mt-3 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Generating...</span>
                  <span className="font-medium text-brand-primary">
                    {progress ? `${Math.round(progress)}%` : 'Processing...'}
                  </span>
                </div>
                <Progress value={progress || 0} className="w-full h-2" />
              </div>
            )}
          </div>

          {/* Content Moderation */}
          <div className="flex items-center justify-between p-3 bg-brand-glass/20 rounded-lg border border-brand-primary/20">
            <div className="flex items-center gap-3">
              <Shield className="w-4 h-4 text-brand-accent" />
              <div>
                <Label className="text-sm font-medium">Content Moderation</Label>
                <p className="text-xs text-muted-foreground">Filter inappropriate content</p>
              </div>
            </div>
            <Switch
              checked={enableModeration}
              onCheckedChange={setEnableModeration}
              disabled={isGenerating}
            />
          </div>

          {/* Cache Expiration */}
          <div className="flex items-center justify-between p-3 bg-brand-glass/20 rounded-lg border border-brand-primary/20">
            <div className="flex items-center gap-3">
              <Settings className="w-4 h-4 text-brand-accent" />
              <div>
                <Label className="text-sm font-medium">Cache Expiration</Label>
                <p className="text-xs text-muted-foreground">Enable cache expiration for generated data</p>
              </div>
            </div>
            <Switch
              checked={cacheExpiration}
              onCheckedChange={setCacheExpiration}
              disabled={isGenerating}
            />
          </div>

          {/* Advanced Settings */}
          <Collapsible open={showAdvanced} onOpenChange={setShowAdvanced}>
            <CollapsibleTrigger asChild>
              <Button
                variant="ghost"
                className="w-full justify-between p-3 h-auto hover:bg-brand-primary/10"
                disabled={isGenerating}
              >
                <div className="flex items-center gap-2">
                  <Settings className="w-4 h-4" />
                  <span>Advanced Settings</span>
                </div>
                <motion.div
                  animate={{ rotate: showAdvanced ? 180 : 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
                    <path d="M2 4l4 4 4-4" />
                  </svg>
                </motion.div>
              </Button>
            </CollapsibleTrigger>

            <CollapsibleContent>
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="space-y-4 pt-4"
              >
                {/* Temperature */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-sm font-medium">Temperature</Label>
                    <Badge variant="secondary" className="bg-brand-secondary/20 text-brand-secondary">
                      {temperature[0].toFixed(1)}
                    </Badge>
                  </div>
                  <Slider
                    value={temperature}
                    onValueChange={setTemperature}
                    max={2.0}
                    min={0.0}
                    step={0.1}
                    disabled={isGenerating}
                    className="[&_[role=slider]]:bg-brand-secondary [&_[role=slider]]:border-brand-secondary"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>Conservative</span>
                    <span>Creative</span>
                  </div>
                </div>

                {/* Max Tokens */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-sm font-medium">Max Tokens</Label>
                    <Badge variant="secondary" className="bg-brand-accent/20 text-brand-accent">
                      {maxTokens[0]}
                    </Badge>
                  </div>
                  <Slider
                    value={maxTokens}
                    onValueChange={setMaxTokens}
                    max={8192}
                    min={512}
                    step={256}
                    disabled={isGenerating}
                    className="[&_[role=slider]]:bg-brand-accent [&_[role=slider]]:border-brand-accent"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>512</span>
                    <span>8192</span>
                  </div>
                </div>

                {/* Top P */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <Label className="text-sm font-medium">Top P</Label>
                    <Badge variant="secondary" className="bg-brand-primary/20 text-brand-primary">
                      {topP[0].toFixed(2)}
                    </Badge>
                  </div>
                  <Slider
                    value={topP}
                    onValueChange={setTopP}
                    max={1.0}
                    min={0.1}
                    step={0.05}
                    disabled={isGenerating}
                    className="[&_[role=slider]]:bg-brand-primary [&_[role=slider]]:border-brand-primary"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>Focused</span>
                    <span>Diverse</span>
                  </div>
                </div>

                <Button
                  variant="outline"
                  size="sm"
                  onClick={resetToDefaults}
                  disabled={isGenerating}
                  className="w-full border-brand-primary/30 hover:bg-brand-primary/10"
                >
                  Reset to Defaults
                </Button>
              </motion.div>
            </CollapsibleContent>
          </Collapsible>

          <Separator className="bg-brand-primary/20" />

          {/* Test Button */}
          <Button
            onClick={() => {
              const testParams: Omit<GenerationRequest, 'input_data'> = {
                count: 3,
                enable_moderation: true,
                temperature: 0.7,
                max_tokens: 2048,
                top_p: 0.9,
                cache_expiration: false,
                user_id: localStorage.getItem('mdg_user_id') || undefined,
                session_id: localStorage.getItem('mdg_session_id') || undefined,
              };
              const sampleData = JSON.stringify([
                {
                  id: 1,
                  name: "John Doe",
                  email: "john@example.com",
                  age: 30,
                  profession: "Software Engineer"
                }
              ], null, 2);
              onGenerate(testParams, sampleData);
            }}
            disabled={isGenerating}
            variant="outline"
            size="sm"
            className="w-full border-brand-primary/30 hover:bg-brand-primary/10"
          >
            Test with Sample Data
          </Button>

          {/* Generate Button */}
          <Button
            onClick={() => onGenerate({
              count: count,
              enable_moderation: enableModeration,
              temperature: temperature[0],
              max_tokens: maxTokens[0],
              top_p: topP[0],
              cache_expiration: cacheExpiration,
              user_id: localStorage.getItem('mdg_user_id') || undefined,
              session_id: localStorage.getItem('mdg_session_id') || undefined,
            })}
            disabled={isGenerating}
            size="lg"
            className="w-full bg-gradient-primary hover:opacity-90 text-white font-semibold h-12 text-base shadow-glow transition-all duration-300"
          >
            <motion.div
              className="flex items-center gap-2"
              animate={isGenerating ? { scale: [1, 1.05, 1] } : {}}
              transition={{ duration: 1, repeat: isGenerating ? Infinity : 0 }}
            >
              {isGenerating ? (
                <>
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-5 h-5" />
                  </motion.div>
                  Generating...
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5" />
                  Generate Mock Data
                </>
              )}
            </motion.div>
          </Button>
        </CardContent>
      </Card>
    </motion.div>
  );
};