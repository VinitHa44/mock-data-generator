import React from 'react';
import { Moon, Sun } from 'lucide-react';
import { useTheme } from '../contexts/ThemeContext';

const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      className={`
        relative inline-flex h-8 w-14 items-center rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-gray-400 focus:ring-offset-2 dark:focus:ring-offset-gray-900 hover:shadow-md
        ${theme === 'dark' ? 'bg-gray-700' : 'bg-gray-200'}
      `}
      aria-label="Toggle theme"
      title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
    >
      <span
        className={`
          inline-block h-6 w-6 transform rounded-full bg-white dark:bg-gray-900 transition-all duration-300 shadow-md
          ${theme === 'dark' ? 'translate-x-7' : 'translate-x-1'}
        `}
      >
        <span className="flex h-full w-full items-center justify-center">
          {theme === 'dark' ? (
            <Moon className="h-3 w-3 text-gray-300" />
          ) : (
            <Sun className="h-3 w-3 text-gray-600" />
          )}
        </span>
      </span>
    </button>
  );
};

export default ThemeToggle; 