import React, { useState, useEffect } from 'react';
import { AlertCircle, FileText, Plus, Trash2 } from 'lucide-react';

interface JsonEditorProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}

const JsonEditor: React.FC<JsonEditorProps> = ({ value, onChange, placeholder }) => {
  const [error, setError] = useState<string>('');
  const [examples, setExamples] = useState<Record<string, any>[]>([]);

  // Validate JSON and update examples
  useEffect(() => {
    if (!value.trim()) {
      setError('');
      setExamples([]);
      return;
    }

    try {
      const parsed = JSON.parse(value);
      if (Array.isArray(parsed)) {
        setExamples(parsed);
        setError('');
      } else {
        setExamples([parsed]);
        setError('');
      }
    } catch (e) {
      setError('Invalid JSON format');
      setExamples([]);
    }
  }, [value]);

  const addExample = () => {
    const newExample = { id: Date.now(), name: 'Sample Name', value: 'Sample Value' };
    const updatedExamples = [...examples, newExample];
    onChange(JSON.stringify(updatedExamples, null, 2));
  };

  const removeExample = (index: number) => {
    const updatedExamples = examples.filter((_, i) => i !== index);
    onChange(JSON.stringify(updatedExamples, null, 2));
  };

  const loadSampleData = () => {
    const sampleData = [
      {
        "id": 1,
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "city": "New York",
        "occupation": "Software Engineer"
      },
      {
        "id": 2,
        "name": "Jane Smith", 
        "email": "jane.smith@example.com",
        "age": 25,
        "city": "San Francisco",
        "occupation": "Product Manager"
      }
    ];
    onChange(JSON.stringify(sampleData, null, 2));
  };

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div className="flex items-center space-x-2">
          <FileText className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          <h3 className="text-lg font-medium text-gray-900 dark:text-white">Example Data</h3>
        </div>
        <div className="flex space-x-2">
          <button
            onClick={loadSampleData}
            className="px-3 py-1 text-sm bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
          >
            Load Sample
          </button>
          <button
            onClick={addExample}
            className="flex items-center space-x-1 px-3 py-1 text-sm bg-gray-900 dark:bg-white text-white dark:text-black rounded hover:bg-gray-800 dark:hover:bg-gray-100 transition-colors"
          >
            <Plus className="h-4 w-4" />
            <span>Add Example</span>
          </button>
        </div>
      </div>

      <div className="relative">
        <textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          placeholder={placeholder || 'Enter JSON array of example objects...'}
          className={`w-full h-64 p-4 border rounded-lg font-mono text-sm resize-y bg-white dark:bg-black text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 ${
            error 
              ? 'border-red-300 dark:border-red-600 focus:border-red-500 dark:focus:border-red-400 focus:ring-red-200 dark:focus:ring-red-800' 
              : 'border-gray-300 dark:border-gray-600 focus:border-gray-500 dark:focus:border-gray-400 focus:ring-gray-200 dark:focus:ring-gray-800'
          } focus:ring-2 focus:outline-none transition-colors`}
        />
        
        {error && (
          <div className="absolute top-2 right-2 flex items-center space-x-1 bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 px-2 py-1 rounded text-sm">
            <AlertCircle className="h-4 w-4" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {examples.length > 0 && (
        <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
            Preview ({examples.length} example{examples.length !== 1 ? 's' : ''})
          </h4>
          <div className="space-y-2 max-h-32 overflow-y-auto">
            {examples.map((example, index) => (
              <div key={index} className="flex items-start justify-between bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                <div className="flex-1">
                  <div className="text-xs text-gray-500 dark:text-gray-400">Example {index + 1}</div>
                  <div className="text-sm text-gray-700 dark:text-gray-300 truncate">
                    {Object.keys(example).join(', ')}
                  </div>
                </div>
                <button
                  onClick={() => removeExample(index)}
                  className="ml-2 p-1 text-gray-400 dark:text-gray-500 hover:text-red-500 dark:hover:text-red-400 transition-colors"
                >
                  <Trash2 className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default JsonEditor; 