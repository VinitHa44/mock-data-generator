import React from 'react';
import { Play, Settings, Loader } from 'lucide-react';

interface GenerationControlsProps {
  count: number;
  onCountChange: (count: number) => void;
  onGenerate: () => void;
  isGenerating: boolean;
  canGenerate: boolean;
}

const GenerationControls: React.FC<GenerationControlsProps> = ({
  count,
  onCountChange,
  onGenerate,
  isGenerating,
  canGenerate
}) => {
  return (
    <div className="bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-800 rounded-lg p-6 space-y-4 transition-colors">
      <div className="flex items-center space-x-2">
        <Settings className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white">Generation Settings</h3>
      </div>

      <div>
        <div className="flex flex-col sm:flex-row items-start sm:items-center space-y-4 sm:space-y-0 sm:space-x-4">
          <div className="flex-grow">
            <label htmlFor="count-slider" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Number of Records: <span className="font-bold text-gray-900 dark:text-white">{count}</span>
            </label>
            <input
              id="count-slider"
              type="range"
              min="1"
              max="500"
              value={count}
              onChange={(e) => onCountChange(parseInt(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              style={{
                background: `linear-gradient(to right, #374151 0%, #374151 ${count/5}%, #e5e7eb ${count/5}%, #e5e7eb 100%)`
              }}
            />
          </div>
          <div className="w-full sm:w-24">
            <label htmlFor="count-input" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 invisible sm:visible">
              Count
            </label>
            <input
              id="count-input"
              type="number"
              min="1"
              max="500"
              value={count}
              onChange={(e) => onCountChange(Math.max(1, Math.min(500, parseInt(e.target.value) || 1)))}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 bg-white dark:bg-black text-gray-900 dark:text-gray-100 rounded focus:border-gray-500 dark:focus:border-gray-400 focus:ring-1 focus:ring-gray-200 dark:focus:ring-gray-800 focus:outline-none transition-colors"
              placeholder="Count"
            />
          </div>
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
          Generate between 1 and 500 mock data records.
        </p>
      </div>

      <div className="border-t border-gray-200 dark:border-gray-800 pt-4">
        <button
          onClick={onGenerate}
          disabled={!canGenerate || isGenerating}
          className={`w-full flex items-center justify-center space-x-2 px-6 py-3 rounded-lg font-medium transition-all duration-200 ${
            canGenerate && !isGenerating
              ? 'bg-gray-900 dark:bg-white hover:bg-gray-800 dark:hover:bg-gray-100 text-white dark:text-black shadow-sm hover:shadow-md'
              : 'bg-gray-300 dark:bg-gray-700 text-gray-500 dark:text-gray-400 cursor-not-allowed'
          }`}
        >
          {isGenerating ? (
            <>
              <Loader className="h-5 w-5 animate-spin" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <Play className="h-5 w-5" />
              <span>Generate Mock Data</span>
            </>
          )}
        </button>
        
        {!canGenerate && !isGenerating && (
          <p className="text-xs text-red-500 dark:text-red-400 mt-2 text-center">
            Please provide valid example data to generate
          </p>
        )}
        
        {isGenerating && (
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-2 text-center">
            AI is analyzing your examples and generating mock data...
          </p>
        )}
      </div>
    </div>
  );
};

export default GenerationControls; 