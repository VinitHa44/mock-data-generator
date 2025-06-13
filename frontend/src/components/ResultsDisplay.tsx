import React, { useState } from 'react';
import { Download, Copy, Eye, EyeOff, CheckCircle, Database, Clock } from 'lucide-react';

interface ResultsDisplayProps {
  data: Record<string, any>[];
  isFromCache: boolean;
  onDownload: () => void;
  onCopy: () => void;
}

const ResultsDisplay: React.FC<ResultsDisplayProps> = ({ 
  data, 
  isFromCache, 
  onDownload, 
  onCopy 
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    await onCopy();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatJsonPreview = (obj: Record<string, any>) => {
    const keys = Object.keys(obj);
    if (keys.length <= 3) {
      return keys.map(key => `${key}: ${JSON.stringify(obj[key])}`).join(', ');
    }
    return keys.slice(0, 3).map(key => `${key}: ${JSON.stringify(obj[key])}`).join(', ') + '...';
  };

  return (
    <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg overflow-hidden transition-colors">
      <div className="bg-gray-50 dark:bg-black px-6 py-4 border-b border-gray-200 dark:border-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Database className="h-5 w-5 text-gray-700 dark:text-gray-300" />
            <div>
              <h3 className="text-lg font-medium text-gray-900 dark:text-white">Generated Results</h3>
              <div className="flex items-center space-x-4 text-sm text-gray-500 dark:text-gray-400">
                <span>{data.length} records generated</span>
                {isFromCache && (
                  <div className="flex items-center space-x-1 text-gray-600 dark:text-gray-400">
                    <Clock className="h-4 w-4" />
                    <span>From cache</span>
                  </div>
                )}
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="flex items-center space-x-1 px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200 transition-colors"
            >
              {isExpanded ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
              <span>{isExpanded ? 'Collapse' : 'Expand'}</span>
            </button>
            <button
              onClick={handleCopy}
              className="flex items-center space-x-1 px-3 py-2 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
            >
              {copied ? <CheckCircle className="h-4 w-4 text-green-600 dark:text-green-400" /> : <Copy className="h-4 w-4" />}
              <span>{copied ? 'Copied!' : 'Copy'}</span>
            </button>
            <button
              onClick={onDownload}
              className="flex items-center space-x-1 px-3 py-2 text-sm bg-gray-900 dark:bg-white text-white dark:text-black rounded hover:bg-gray-800 dark:hover:bg-gray-100 transition-colors"
            >
              <Download className="h-4 w-4" />
              <span>Download JSON</span>
            </button>
          </div>
        </div>
      </div>

      <div className="p-6">
        {isExpanded ? (
          <div className="space-y-4">
            <pre className="bg-gray-50 dark:bg-black text-gray-900 dark:text-gray-100 p-4 rounded-lg overflow-x-auto text-sm font-mono max-h-96 overflow-y-auto border border-gray-200 dark:border-gray-700">
              {JSON.stringify(data, null, 2)}
            </pre>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="text-sm text-gray-600 dark:text-gray-400 mb-3">Preview (showing first 3 records)</div>
            {data.slice(0, 3).map((item, index) => (
              <div key={index} className="bg-gray-50 dark:bg-black p-3 rounded border border-gray-200 dark:border-gray-700">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Record {index + 1}</div>
                    <div className="text-sm text-gray-700 dark:text-gray-300 font-mono break-all">
                      {formatJsonPreview(item)}
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {data.length > 3 && (
              <div className="text-center py-2">
                <button
                  onClick={() => setIsExpanded(true)}
                  className="text-sm text-gray-700 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white transition-colors underline"
                >
                  Show all {data.length} records
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ResultsDisplay; 