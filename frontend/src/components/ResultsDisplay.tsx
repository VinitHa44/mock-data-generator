import React, { useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Editor from '@monaco-editor/react';
import { 
  Download, 
  Copy, 
  Search, 
  Filter, 
  Grid, 
  List, 
  Eye, 
  MoreHorizontal,
  Check,
  ExternalLink,
  Hash,
  Clock,
  Sparkles,
  Code
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Progress } from '@/components/ui/progress';
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger 
} from '@/components/ui/dropdown-menu';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import toast from 'react-hot-toast';
import type { GenerateMockDataResponse } from '@/types/api';

interface ResultsDisplayProps {
  results: GenerateMockDataResponse | null;
  isLoading: boolean;
}

export const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ 
  results, 
  isLoading
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [viewMode, setViewMode] = useState<'grid' | 'list' | 'json'>('json');
  const [copied, setCopied] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<number | null>(null);
  const [isDetailModalOpen, setIsDetailModalOpen] = useState(false);

  const filteredData = useMemo(() => {
    if (!results?.data || !searchQuery) return results?.data || [];
    
    return results.data.filter(record =>
      JSON.stringify(record).toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [results?.data, searchQuery]);

  const handleCopyAll = async () => {
    if (!results?.data) return;
    
    try {
      await navigator.clipboard.writeText(JSON.stringify(results.data, null, 2));
      setCopied(true);
      toast.success('All data copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast.error('Failed to copy data');
    }
  };

  const handleCopyRecord = async (record: Record<string, unknown>, index: number) => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(record, null, 2));
      toast.success(`Record ${index + 1} copied`);
    } catch (error) {
      toast.error('Failed to copy record');
    }
  };

  const handleDownload = (format: 'json' | 'csv') => {
    if (!results?.data) return;

    try {
      let content: string;
      let filename: string;
      let mimeType: string;

      if (format === 'json') {
        content = JSON.stringify(results.data, null, 2);
        filename = `mock-data-${Date.now()}.json`;
        mimeType = 'application/json';
      } else {
        // Convert to CSV
        const headers = Object.keys(results.data[0] || {});
        const csvRows = [
          headers.join(','),
          ...results.data.map(record =>
            headers.map(header => {
              const value = record[header];
              const stringValue = typeof value === 'string' ? value : JSON.stringify(value);
              return `"${stringValue.replace(/"/g, '""')}"`;
            }).join(',')
          )
        ];
        content = csvRows.join('\n');
        filename = `mock-data-${Date.now()}.csv`;
        mimeType = 'text/csv';
      }

      const blob = new Blob([content], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      toast.success(`Downloaded as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Failed to download file');
    }
  };

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-4"
      >
        <Card className="bg-gradient-to-br from-background via-background to-brand-glass/20 border-brand-primary/20">
          <CardContent className="p-8">
            <div className="flex flex-col items-center justify-center space-y-4">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                className="w-12 h-12 border-4 border-brand-primary/20 border-t-brand-primary rounded-full"
              />
              <div className="text-center">
                <h3 className="font-semibold text-lg mb-2">Generating Your Data</h3>
                <p className="text-muted-foreground">
                  Our AI is crafting high-quality mock data for you...
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  if (!results) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-4"
      >
        <Card className="bg-gradient-to-br from-background via-background to-brand-glass/20 border-brand-primary/20">
          <CardContent className="p-8">
            <div className="text-center space-y-4">
              <div className="w-16 h-16 mx-auto bg-brand-primary/10 rounded-full flex items-center justify-center">
                <Sparkles className="w-8 h-8 text-brand-primary" />
              </div>
              <div>
                <h3 className="font-semibold text-lg mb-2">Ready to Generate</h3>
                <p className="text-muted-foreground">
                  Add your JSON template and click generate to create mock data.
                </p>
              </div>
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
      transition={{ duration: 0.3, delay: 0.2 }}
      className="space-y-6"
    >
      {/* Results Header */}
      <Card className="bg-gradient-to-br from-background via-background to-brand-glass/20 border-brand-primary/20 shadow-elegant">
        <CardHeader className="pb-4">
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2 text-lg font-semibold bg-gradient-primary bg-clip-text text-transparent">
              <Hash className="w-5 h-5 text-brand-primary" />
              Generated Results
            </CardTitle>
            
            {/* Cache Info */}
            {results.cacheInfo && (
              <div className="flex items-center gap-2">
                <Badge className="bg-brand-accent text-white">
                  {results.data.length} Records
                </Badge>
                {results.usedFromCache && (
                  <Badge variant="secondary" className="bg-brand-primary/20 text-brand-primary">
                    <Sparkles className="w-3 h-3 mr-1" />
                    Cached
                  </Badge>
                )}
              </div>
            )}
          </div>
          
          {/* Cache Info */}
          {results.cacheInfo && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
              <div className="bg-brand-glass/20 p-3 rounded-lg border border-brand-primary/20">
                <div className="text-sm text-muted-foreground">Cache Hit Type</div>
                <div className="font-semibold text-brand-primary capitalize">
                  {results.cacheInfo.cacheHitType}
                </div>
              </div>
              <div className="bg-brand-glass/20 p-3 rounded-lg border border-brand-primary/20">
                <div className="text-sm text-muted-foreground">Cached Records</div>
                <div className="font-semibold text-brand-secondary">
                  {results.cacheInfo.cachedCount}
                </div>
              </div>
              <div className="bg-brand-glass/20 p-3 rounded-lg border border-brand-primary/20">
                <div className="text-sm text-muted-foreground">Generated Records</div>
                <div className="font-semibold text-brand-accent">
                  {results.cacheInfo.generatedCount}
                </div>
              </div>
              <div className="bg-brand-glass/20 p-3 rounded-lg border border-brand-primary/20">
                <div className="text-sm text-muted-foreground">Total Records</div>
                <div className="font-semibold text-brand-primary">
                  {results.cacheInfo.totalCount}
                </div>
              </div>
            </div>
          )}
        </CardHeader>

        <CardContent className="pt-0">
          {/* Controls */}
          <div className="flex flex-col sm:flex-row gap-4 justify-between items-start sm:items-center mb-6">
            <div className="flex items-center gap-2 flex-1">
              <div className="relative flex-1 max-w-md">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search records..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 border-brand-primary/30 focus:border-brand-primary"
                />
              </div>
              <div className="text-sm text-muted-foreground">
                {filteredData.length} of {results.data.length}
              </div>
            </div>
            
            <div className="flex items-center gap-2">
              <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'grid' | 'list' | 'json')}>
                <TabsList className="bg-brand-glass/20 border border-brand-primary/20">
                <TabsTrigger value="json" className="data-[state=active]:bg-brand-primary data-[state=active]:text-white">
                    <Code className="w-4 h-4" />
                  </TabsTrigger>
                  <TabsTrigger value="grid" className="data-[state=active]:bg-brand-primary data-[state=active]:text-white">
                    <Grid className="w-4 h-4" />
                  </TabsTrigger>
                  <TabsTrigger value="list" className="data-[state=active]:bg-brand-primary data-[state=active]:text-white">
                    <List className="w-4 h-4" />
                  </TabsTrigger>
                </TabsList>
              </Tabs>
              
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopyAll}
                className="border-brand-primary/30 hover:bg-brand-primary/10"
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              </Button>
              
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" size="sm" className="border-brand-primary/30 hover:bg-brand-primary/10">
                    <Download className="w-4 h-4 mr-2" />
                    Export
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={() => handleDownload('json')}>
                    Download JSON
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={() => handleDownload('csv')}>
                    Download CSV
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          </div>

          {/* Data Display */}
          <AnimatePresence mode="wait">
            {viewMode === 'grid' ? (
              <motion.div
                key="grid"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4"
              >
                {filteredData.map((record, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: index * 0.05 }}
                    className="bg-brand-glass/20 border border-brand-primary/20 rounded-lg p-4 hover:shadow-glow/20 transition-all duration-300 group"
                  >
                    <div className="flex items-center justify-between mb-3">
                      <Badge variant="secondary" className="bg-brand-primary/20 text-brand-primary">
                        Record {index + 1}
                      </Badge>
                      
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm" className="opacity-0 group-hover:opacity-100 transition-opacity">
                            <MoreHorizontal className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem onClick={() => handleCopyRecord(record, index)}>
                            <Copy className="w-4 h-4 mr-2" />
                            Copy Record
                          </DropdownMenuItem>
                          <DropdownMenuItem onClick={() => {
                            setSelectedRecord(index);
                            setIsDetailModalOpen(true);
                          }}>
                            <Eye className="w-4 h-4 mr-2" />
                            View Details
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                    
                    <div className="space-y-2">
                      {Object.entries(record).slice(0, 3).map(([key, value]) => (
                        <div key={key} className="text-sm">
                          <span className="font-medium text-brand-primary">{key}:</span>
                          <span className="ml-2 text-muted-foreground">
                            {typeof value === 'string' && value.length > 30
                              ? `${value.substring(0, 30)}...`
                              : String(value)
                            }
                          </span>
                        </div>
                      ))}
                      {Object.keys(record).length > 3 && (
                        <div className="text-xs text-muted-foreground">
                          +{Object.keys(record).length - 3} more fields
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </motion.div>
            ) : viewMode === 'list' ? (
              <motion.div
                key="list"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-2"
              >
                {filteredData.map((record, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.02 }}
                    className="bg-brand-glass/20 border border-brand-primary/20 rounded-lg p-4 hover:shadow-glow/20 transition-all duration-300"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <Badge variant="secondary" className="bg-brand-primary/20 text-brand-primary">
                        Record {index + 1}
                      </Badge>
                      <div className="flex items-center gap-2">
                        <Button 
                          variant="ghost" 
                          size="sm"
                          onClick={() => handleCopyRecord(record, index)}
                        >
                          <Copy className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>
                    
                    <pre className="text-xs bg-brand-surface/5 p-3 rounded border border-brand-primary/10 overflow-x-auto">
                      {JSON.stringify(record, null, 2)}
                    </pre>
                  </motion.div>
                ))}
              </motion.div>
            ) : (
              <motion.div
                key="json"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-4"
              >
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-medium text-muted-foreground">
                    Generated JSON Data
                    {searchQuery && (
                      <span className="ml-2 text-xs text-muted-foreground">
                        (Filtered: {filteredData.length} of {results.data.length})
                      </span>
                    )}
                  </h4>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      navigator.clipboard.writeText(JSON.stringify(filteredData, null, 2));
                      toast.success('Filtered JSON copied to clipboard');
                    }}
                    className="border-brand-primary/30 hover:bg-brand-primary/10"
                  >
                    <Copy className="w-4 h-4 mr-2" />
                    Copy JSON
                  </Button>
                </div>
                <div className="border border-brand-primary/20 rounded-lg overflow-hidden bg-brand-surface/5">
                  <Editor
                    height="400px"
                    defaultLanguage="json"
                    value={JSON.stringify(filteredData, null, 2)}
                    options={{
                      readOnly: true,
                      wordWrap: 'on',
                      lineNumbers: 'on',
                      folding: true,
                      bracketPairColorization: { enabled: true },
                      autoIndent: 'advanced',
                      tabSize: 2,
                      insertSpaces: true,
                      fontSize: 14,
                      fontFamily: 'JetBrains Mono, Consolas, monospace',
                      theme: 'vs-dark',
                    }}
                    theme="vs-dark"
                  />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {filteredData.length === 0 && searchQuery && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-8"
            >
              <div className="text-muted-foreground">No records match your search.</div>
            </motion.div>
          )}
        </CardContent>
      </Card>

      {/* Record Details Modal */}
      <Dialog open={isDetailModalOpen} onOpenChange={setIsDetailModalOpen}>
        <DialogContent className="max-w-4xl max-h-[80vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Eye className="w-5 h-5 text-brand-primary" />
              Record Details
              {selectedRecord !== null && (
                <Badge variant="secondary" className="bg-brand-primary/20 text-brand-primary">
                  Record {selectedRecord + 1}
                </Badge>
              )}
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-4">
            {selectedRecord !== null && results?.data[selectedRecord] && (
              <div className="border border-brand-primary/20 rounded-lg overflow-hidden bg-brand-surface/5">
                <Editor
                  height="400px"
                  defaultLanguage="json"
                  value={JSON.stringify(results.data[selectedRecord], null, 2)}
                  options={{
                    readOnly: true,
                    wordWrap: 'on',
                    lineNumbers: 'on',
                    folding: true,
                    bracketPairColorization: { enabled: true },
                    autoIndent: 'advanced',
                    tabSize: 2,
                    insertSpaces: true,
                    fontSize: 14,
                    fontFamily: 'JetBrains Mono, Consolas, monospace',
                    theme: 'vs-dark',
                  }}
                  theme="vs-dark"
                />
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
};