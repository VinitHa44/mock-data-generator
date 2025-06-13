import React from 'react';
import { Database, Zap } from 'lucide-react';
import ThemeToggle from './ThemeToggle';

const Header: React.FC = () => {
  return (
    <header className="bg-white dark:bg-black shadow-sm border-b border-gray-200 dark:border-gray-800 transition-colors duration-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-6">
          <div className="flex items-center space-x-3">
            <div className="bg-gray-900 dark:bg-white p-2 rounded-lg">
              <Database className="h-8 w-8 text-white dark:text-black" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Mock Data Generator</h1>
              <p className="text-sm text-gray-600 dark:text-gray-400">AI-powered synthetic data creation</p>
            </div>
          </div>
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2 bg-gray-100 dark:bg-gray-800 px-3 py-2 rounded-lg">
              <Zap className="h-5 w-5 text-gray-700 dark:text-gray-300" />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">LLM Powered</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-xs text-gray-500 dark:text-gray-400 hidden sm:block">Theme</span>
              <ThemeToggle />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header; 