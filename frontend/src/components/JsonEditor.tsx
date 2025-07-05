import React, { useState, useCallback, useEffect } from 'react';
import Editor from '@monaco-editor/react';
import { motion } from 'framer-motion';
import { Copy, Download, Upload, Sparkles, Check, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import toast from 'react-hot-toast';

interface JsonEditorProps {
  value: string;
  onChange: (value: string) => void;
  onValidationChange: (isValid: boolean) => void;
  placeholder?: string;
  height?: number;
  readOnly?: boolean;
  title?: string;
}

export const JsonEditor: React.FC<JsonEditorProps> = ({
  value,
  onChange,
  onValidationChange,
  placeholder = '[\n  {\n    "name": "John Doe",\n    "email": "john@example.com",\n    "age": 28\n  }\n]',
  height = 300,
  readOnly = false,
  title = "JSON Data"
}) => {
  const [isValid, setIsValid] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const validateJson = useCallback((jsonString: string) => {
    if (!jsonString.trim()) {
      setIsValid(true);
      setError(null);
      onValidationChange(true);
      return;
    }

    try {
      const parsed = JSON.parse(jsonString);
      // Accept both arrays and single objects, but ensure they contain objects
      if (Array.isArray(parsed)) {
        if (parsed.length === 0) {
          setIsValid(false);
          setError('JSON array must contain at least one object');
          onValidationChange(false);
        } else if (parsed.every(item => typeof item === 'object' && item !== null)) {
          setIsValid(true);
          setError(null);
          onValidationChange(true);
        } else {
          setIsValid(false);
          setError('JSON array must contain only objects');
          onValidationChange(false);
        }
      } else if (typeof parsed === 'object' && parsed !== null) {
        // Single object is also valid
        setIsValid(true);
        setError(null);
        onValidationChange(true);
      } else {
        setIsValid(false);
        setError('JSON must be an object or array of objects');
        onValidationChange(false);
      }
    } catch (e) {
      setIsValid(false);
      setError(e instanceof Error ? e.message : 'Invalid JSON format');
      onValidationChange(false);
    }
  }, [onValidationChange]);

  // Validate JSON when value changes
  useEffect(() => {
    validateJson(value);
  }, [value, validateJson]);

  const handleEditorChange = useCallback((newValue: string | undefined) => {
    const value = newValue || '';
    onChange(value);
    // Validation will be triggered by useEffect
  }, [onChange]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      toast.success('Copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    } catch (error) {
      toast.error('Failed to copy');
    }
  };

  const handleDownload = () => {
    try {
      const blob = new Blob([value], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${title.toLowerCase().replace(/\s+/g, '-')}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      toast.success('File downloaded');
    } catch (error) {
      toast.error('Failed to download');
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== 'application/json' && !file.name.endsWith('.json')) {
      toast.error('Please select a JSON file');
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target?.result as string;
      onChange(content);
      validateJson(content);
      toast.success('File loaded successfully');
    };
    reader.onerror = () => {
      toast.error('Failed to read file');
    };
    reader.readAsText(file);
    
    // Reset input
    event.target.value = '';
  };

  const formatJson = () => {
    try {
      const parsed = JSON.parse(value);
      const formatted = JSON.stringify(parsed, null, 2);
      onChange(formatted);
      toast.success('JSON formatted');
    } catch (error) {
      toast.error('Cannot format invalid JSON');
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
    >
      <Card className="overflow-hidden bg-gradient-to-br from-background via-background to-brand-glass/20 border-brand-primary/20 shadow-elegant">
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CardTitle className="text-lg font-semibold bg-gradient-primary bg-clip-text text-transparent">
                {title}
              </CardTitle>
              {!isValid && (
                <Badge variant="destructive" className="animate-pulse">
                  <AlertCircle className="w-3 h-3 mr-1" />
                  Invalid
                </Badge>
              )}
              {isValid && value.trim() && (
                <Badge className="bg-brand-accent text-white">
                  <Check className="w-3 h-3 mr-1" />
                  Valid
                </Badge>
              )}
            </div>
            
            <div className="flex items-center gap-2">
              {!readOnly && (
                <>
                  <input
                    type="file"
                    accept=".json,application/json"
                    onChange={handleFileUpload}
                    className="hidden"
                    id={`file-upload-${title}`}
                  />
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => document.getElementById(`file-upload-${title}`)?.click()}
                    className="h-8 px-3 border-brand-primary/30 hover:bg-brand-primary/10"
                  >
                    <Upload className="w-3 h-3" />
                  </Button>
                  
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={formatJson}
                    disabled={!isValid}
                    className="h-8 px-3 border-brand-primary/30 hover:bg-brand-primary/10"
                  >
                    <Sparkles className="w-3 h-3" />
                  </Button>
                </>
              )}
              
              <Button
                variant="outline"
                size="sm"
                onClick={handleCopy}
                disabled={!value.trim()}
                className="h-8 px-3 border-brand-primary/30 hover:bg-brand-primary/10"
              >
                {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
              </Button>
              
              <Button
                variant="outline"
                size="sm"
                onClick={handleDownload}
                disabled={!value.trim() || !isValid}
                className="h-8 px-3 border-brand-primary/30 hover:bg-brand-primary/10"
              >
                <Download className="w-3 h-3" />
              </Button>
            </div>
          </div>
          
          {error && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              className="text-sm text-destructive bg-destructive/10 p-2 rounded border border-destructive/20"
            >
              {error}
            </motion.div>
          )}
        </CardHeader>
        
        <CardContent className="p-0">
          <div className="border border-brand-primary/20 rounded-lg m-4 overflow-hidden bg-brand-surface/5 backdrop-blur-sm">
            <Editor
              height={height}
              defaultLanguage="json"
              value={value}
              onChange={handleEditorChange}
              options={{
                readOnly,
                minimap: { enabled: false },
                scrollBeyondLastLine: false,
                wordWrap: 'on',
                lineNumbers: 'on',
                folding: true,
                bracketPairColorization: { enabled: true },
                autoIndent: 'advanced',
                formatOnPaste: true,
                formatOnType: true,
                tabSize: 2,
                insertSpaces: true,
                fontSize: 14,
                fontFamily: 'JetBrains Mono, Consolas, monospace',
                theme: 'vs-dark',
                padding: { top: 16, bottom: 16 },
                smoothScrolling: true,
                cursorBlinking: 'smooth',
                renderLineHighlight: 'all',
                selectionHighlight: false,
                occurrencesHighlight: false,
              }}
              theme="vs-dark"
              loading={
                <div className="flex items-center justify-center h-full">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-brand-primary"></div>
                </div>
              }
            />
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};