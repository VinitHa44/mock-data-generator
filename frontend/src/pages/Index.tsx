import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import { HeroSection } from '@/components/HeroSection';
import { JsonEditor } from '@/components/JsonEditor';
import { GenerationControls } from '@/components/GenerationControls';
import { ResultsDisplay } from '@/components/ResultsDisplay';
import { MockDataGeneratorAPI } from '@/services/api';
import type { GenerationRequest, GenerateMockDataResponse, Template } from '@/types/api';
import { getErrorMessage } from '@/lib/utils';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { 
  Database, 
  Sparkles, 
  Activity, 
  BookOpen, 
  Zap,
  ChevronDown,
  Star,
  Copy,
  ExternalLink
} from 'lucide-react';
import toast from 'react-hot-toast';

const Index = () => {
  const [showApp, setShowApp] = useState(false);
  const [jsonInput, setJsonInput] = useState(`[
  {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "age": 28,
    "profession": "Software Engineer",
    "location": "San Francisco, CA",
    "salary": 120000,
    "skills": ["JavaScript", "React", "Node.js"],
    "isActive": true
  }
]`);
  const [isJsonValid, setIsJsonValid] = useState(true);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationResults, setGenerationResults] = useState<GenerateMockDataResponse | null>(null);
  const [generationProgress, setGenerationProgress] = useState(0);
  const [templates, setTemplates] = useState<Template[]>([]);
  const [selectedTemplate, setSelectedTemplate] = useState<Template | null>(null);

  // Health check on app load
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await MockDataGeneratorAPI.getSystemHealth();
        console.log('Backend is healthy');
      } catch (error) {
        console.warn('Backend health check failed:', error);
        // The specific error message is already handled by the API interceptor
        // This is just a fallback for the health check
        toast.error(getErrorMessage(error));
      }
    };
    
    checkBackendHealth();
  }, []);

  // Initialize session
  useEffect(() => {
    if (!localStorage.getItem('mdg_session_id')) {
      localStorage.setItem('mdg_session_id', `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    }
    if (!localStorage.getItem('mdg_user_id')) {
      localStorage.setItem('mdg_user_id', `user_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
    }

    // Load default templates (no API call since endpoint doesn't exist)
    setTemplates(MockDataGeneratorAPI.getDefaultTemplates());
  }, []);

  const handleGetStarted = () => {
    setShowApp(true);
    setTimeout(() => {
      document.getElementById('app-interface')?.scrollIntoView({ 
        behavior: 'smooth' 
      });
    }, 100);
  };

  const handleGenerate = async (params: Omit<GenerationRequest, 'input_data'>, sampleData?: string) => {
    const inputToUse = sampleData || jsonInput;
    
    if (!inputToUse.trim()) {
      toast.error('Please provide JSON input data');
      return;
    }
    
    if (!sampleData && !isJsonValid) {
      toast.error('Please provide valid JSON input data');
      return;
    }

    try {
      const inputData = JSON.parse(inputToUse);
      
      // ** NEW: Validate that input is a non-empty array of objects **
      if (!Array.isArray(inputData) || inputData.length === 0) {
        toast.error('Input must be a non-empty array of objects.');
        return;
      }
      
      // Ensure all items in the array are objects
      if (inputData.some(item => typeof item !== 'object' || item === null || Array.isArray(item))) {
          toast.error('All items in the array must be objects.');
          return;
      }

      setIsGenerating(true);
      setGenerationProgress(0);
      
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setGenerationProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 500);
      
      // Use the same API call format as working frontend
      const response = await MockDataGeneratorAPI.generateMockData(
        inputData, 
        params.count,
        {
          enable_moderation: params.enable_moderation,
          temperature: params.temperature,
          max_tokens: params.max_tokens,
          top_p: params.top_p,
          cache_expiration: params.cache_expiration
        }
      );
      
      clearInterval(progressInterval);
      setGenerationProgress(100);
      
      setGenerationResults(response);
      
      // Create appropriate toast message based on cache info
      let toastMessage = `Generated ${response.data.length} records successfully`;
      
      if (response.cacheInfo) {
        const { cachedCount, generatedCount, cacheHitType } = response.cacheInfo;
        
        switch (cacheHitType) {
          case 'full':
            toastMessage = `Retrieved ${cachedCount} records from cache`;
            break;
          case 'partial':
            toastMessage = `Retrieved ${cachedCount} from cache + generated ${generatedCount} new records`;
            break;
          case 'none':
            toastMessage = `Generated ${generatedCount} new records`;
            break;
        }
      } else if (response.usedFromCache) {
        toastMessage = `Retrieved ${response.data.length} records from cache`;
      }
      
      toast.success(toastMessage);
      
      // Scroll to results
      setTimeout(() => {
        document.getElementById('results-section')?.scrollIntoView({ 
          behavior: 'smooth' 
        });
      }, 500);
      
    } catch (error: unknown) {
      console.error('Generation failed:', error);
      // The specific error message is already handled by the API interceptor
      // This is just a fallback for generation-specific errors
      toast.error(getErrorMessage(error));
    } finally {
      setIsGenerating(false);
      setGenerationProgress(0);
    }
  };

  if (!showApp) {
    return (
      <>
        <HeroSection onGetStarted={handleGetStarted} />
        
        {/* Features Section */}
        <section id="features" className="py-20 bg-gradient-to-br from-brand-glass/5 to-background">
          <div className="max-w-6xl mx-auto px-6">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8 }}
              viewport={{ once: true }}
              className="text-center mb-16"
            >
              <h2 className="text-4xl font-bold mb-4 bg-gradient-primary bg-clip-text text-transparent">
                Powerful Features
              </h2>
              <p className="text-xl text-muted-foreground max-w-3xl mx-auto">
                Everything you need to generate high-quality mock data with AI precision.
              </p>
            </motion.div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {[
                {
                  icon: Sparkles,
                  title: 'AI-Powered Generation',
                  description: 'Advanced language models create realistic, contextually appropriate data that matches your examples perfectly.'
                },
                {
                  icon: Zap,
                  title: 'Lightning Fast',
                  description: 'Generate thousands of records in seconds with our optimized local processing pipeline.'
                },
                {
                  icon: Database,
                  title: 'Flexible Data Types',
                  description: 'Support for any JSON structure - from simple user profiles to complex nested business objects.'
                },
                {
                  icon: Activity,
                  title: 'Real-time Monitoring',
                  description: 'Track system performance, queue status, and generation metrics in real-time.'
                },
                {
                  icon: BookOpen,
                  title: 'Template Library',
                  description: 'Pre-built templates for common use cases - users, products, transactions, and more.'
                },
                {
                  icon: Star,
                  title: 'Enterprise Ready',
                  description: 'Built for scale with distributed architecture, caching, and comprehensive monitoring.'
                }
              ].map((feature, index) => (
                <motion.div
                  key={feature.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: index * 0.1 }}
                  viewport={{ once: true }}
                  className="group"
                >
                  <Card className="h-full bg-glass backdrop-blur-glass border-brand-primary/20 shadow-glass hover:shadow-glow/20 transition-all duration-300">
                    <CardContent className="p-6">
                      <div className="w-12 h-12 bg-gradient-primary rounded-lg flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300">
                        <feature.icon className="w-6 h-6 text-white" />
                      </div>
                      <h3 className="text-xl font-semibold mb-2 text-brand-primary">
                        {feature.title}
                      </h3>
                      <p className="text-muted-foreground">
                        {feature.description}
                      </p>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
              viewport={{ once: true }}
              className="text-center mt-16"
            >
              <Button
                onClick={handleGetStarted}
                size="lg"
                className="bg-gradient-primary hover:opacity-90 text-white font-semibold px-8 py-6 text-lg shadow-glow"
              >
                <Zap className="w-5 h-5 mr-2" />
                Start Generating Data
              </Button>
            </motion.div>
          </div>
        </section>
        
        <Toaster 
          position="top-right"
          toastOptions={{
            className: 'bg-background border border-brand-primary/20 text-foreground',
            duration: 4000,
          }}
        />
      </>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-brand-glass/5 to-brand-primary/5">
      <div id="app-interface" className="max-w-7xl mx-auto px-6 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="flex items-center justify-between mb-6">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                Mock Data Generator
              </h1>
              <p className="text-muted-foreground mt-1">
                AI-powered data generation for your development needs
              </p>
            </div>
            
            <div className="flex items-center gap-4">
              <Badge className="bg-brand-accent text-white">
                <Activity className="w-3 h-3 mr-1" />
                Live System
              </Badge>
              <Button
                variant="outline"
                onClick={() => setShowApp(false)}
                className="border-brand-primary/30 hover:bg-brand-primary/10"
              >
                Back to Home
              </Button>
            </div>
          </div>
          
          <Separator className="bg-brand-primary/20" />
        </motion.div>

        <Tabs defaultValue="generator" className="space-y-6">

          <TabsContent value="generator" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* JSON Editor */}
              <div className="space-y-4">
                <JsonEditor
                  value={jsonInput}
                  onChange={setJsonInput}
                  onValidationChange={setIsJsonValid}
                  title="Input JSON Template"
                  height={400}
                  placeholder={`[
  {
    "id": 1,
    "name": "John Doe",
    "email": "john@example.com",
    "age": 28,
    "profession": "Software Engineer",
    "location": "San Francisco, CA",
    "salary": 120000,
    "skills": ["JavaScript", "React", "Node.js"],
    "isActive": true
  }
]`}
                />
                
                {selectedTemplate && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="bg-brand-glass/20 border border-brand-primary/20 rounded-lg p-3"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge className="bg-brand-accent text-white">
                          Template: {selectedTemplate.name}
                        </Badge>
                        <span className="text-sm text-muted-foreground">
                          {selectedTemplate.category}
                        </span>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedTemplate(null)}
                      >
                        Clear
                      </Button>
                    </div>
                  </motion.div>
                )}
              </div>

              {/* Generation Controls */}
              <GenerationControls
                onGenerate={handleGenerate}
                isGenerating={isGenerating}
                progress={generationProgress}
              />
            </div>

            {/* Results Section */}
            <div id="results-section">
              <ResultsDisplay
                results={generationResults}
                isLoading={isGenerating}
              />
            </div>
          </TabsContent>
        </Tabs>
      </div>
      
      <Toaster 
        position="top-right"
        toastOptions={{
          className: 'bg-background border border-brand-primary/20 text-foreground',
          duration: 4000,
        }}
      />
    </div>
  );
};

export default Index;
