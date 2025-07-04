import React, { useState, useEffect } from 'react';
import { Toaster, toast } from 'react-hot-toast';
import Header from './components/Header';
import JsonEditor from './components/JsonEditor';
import GenerationControls from './components/GenerationControls';
import ResultsDisplay from './components/ResultsDisplay';
import { mockDataApi } from './services/api';
import { ThemeProvider } from './contexts/ThemeContext';
import { APIResponse } from './types/api';

function App() {
  const [exampleData, setExampleData] = useState<string>('');
  const [count, setCount] = useState<number>(10);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [response, setResponse] = useState<APIResponse | null>(null);

  // Check if we can generate (valid JSON with at least one example)
  const canGenerate = () => {
    if (!exampleData.trim()) return false;
    try {
      const parsed = JSON.parse(exampleData);
      return Array.isArray(parsed) ? parsed.length > 0 : true;
    } catch {
      return false;
    }
  };

  const handleGenerate = async () => {
    if (!canGenerate()) {
      toast.error('Please provide valid example data');
      return;
    }

    setIsGenerating(true);
    setResponse(null);
    
    try {
      const examples = JSON.parse(exampleData);
      const exampleArray = Array.isArray(examples) ? examples : [examples];
      
      toast.loading('Generating mock data...', { id: 'generation' });
      
      const apiResponse = await mockDataApi.generateMockData(exampleArray, { count });
      
      setResponse(apiResponse);
      
      // Create appropriate toast message based on cache info
      let toastMessage = `Generated ${apiResponse.data.length} records successfully`;
      
      if (apiResponse.cacheInfo) {
        const { cachedCount, generatedCount, cacheHitType } = apiResponse.cacheInfo;
        
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
      } else if (apiResponse.usedFromCache) {
        toastMessage = `Retrieved ${apiResponse.data.length} records from cache`;
      }
      
      toast.success(toastMessage, { id: 'generation' });
    } catch (error: any) {
      console.error('Generation error:', error);
      let errorMessage = 'Failed to generate mock data';
      
      if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      toast.error(errorMessage, { id: 'generation' });
    } finally {
      setIsGenerating(false);
    }
  };

  const handleDownload = () => {
    if (!response?.data) return;
    
    const dataStr = JSON.stringify(response.data, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    
    const link = document.createElement('a');
    link.href = url;
    link.download = `mock-data-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    
    toast.success('Downloaded successfully');
  };

  const handleCopy = async () => {
    if (!response?.data) return;
    
    try {
      await navigator.clipboard.writeText(JSON.stringify(response.data, null, 2));
      toast.success('Copied to clipboard');
    } catch (error) {
      toast.error('Failed to copy to clipboard');
    }
  };

  // Health check on app load
  useEffect(() => {
    const checkBackendHealth = async () => {
      try {
        await mockDataApi.healthCheck();
        console.log('Backend is healthy');
      } catch (error) {
        console.warn('Backend health check failed:', error);
        toast.error('Unable to connect to backend service');
      }
    };
    
    checkBackendHealth();
  }, []);

  return (
    <ThemeProvider>
      <div className="min-h-screen bg-gray-50 dark:bg-black transition-colors">
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#363636',
              color: '#fff',
            },
          }}
        />
        
        <Header />
        
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Input Section */}
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 transition-colors">
                <JsonEditor
                  value={exampleData}
                  onChange={setExampleData}
                  placeholder={`Enter your example data as JSON array:\n[\n  {\n    "id": 1,\n    "name": "John Doe",\n    "email": "john@example.com"\n  }\n]`}
                />
              </div>
              
              <ResultsDisplay
                response={response}
                isLoading={isGenerating}
                onDownload={handleDownload}
                onCopy={handleCopy}
              />
            </div>
            
            {/* Controls Section */}
            <div className="space-y-6">
              <GenerationControls
                count={count}
                onCountChange={setCount}
                onGenerate={handleGenerate}
                isGenerating={isGenerating}
                canGenerate={canGenerate()}
              />
              
              {/* Info Section */}
              <div className="bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 transition-colors">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">How it Works</h3>
                <ol className="list-decimal list-inside space-y-2 text-gray-700 dark:text-gray-300">
                  <li>• Paste your example JSON into the editor</li>
                  <li>• Generate up to 500 similar records</li>
                  <li>• Get your structured mock data instantly</li>
                </ol>
              </div>
              
              {/* Features */}
              <div className="bg-gray-100 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 transition-colors">
                <h4 className="text-sm font-medium text-gray-900 dark:text-white mb-2">Features</h4>
                <ul className="text-xs text-gray-700 dark:text-gray-300 space-y-1">
                  <li>• ✨ AI-powered data generation</li>
                  <li>• 🎯 Pattern-aware field generation</li>
                  <li>• 🔄 Intelligent caching system</li>
                  <li>• 🖼️ Image URL enrichment</li>
                  <li>• 📊 Structured JSON output</li>
                </ul>
              </div>
            </div>
          </div>
        </main>
      </div>
    </ThemeProvider>
  );
}

export default App;
