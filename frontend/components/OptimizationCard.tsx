'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Lightbulb, 
  ChevronRight, 
  Copy, 
  CheckCircle,
  TrendingUp,
  Clock,
  Star
} from 'lucide-react';
import { OptimizationCardProps } from '@/types';

const OptimizationCard: React.FC<OptimizationCardProps> = ({
  suggestion,
  index,
  onExpand,
  isExpanded = false,
}) => {
  const [copied, setCopied] = useState(false);

  const getPriorityColor = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high':
        return 'bg-error-100 text-error-800 border-error-200';
      case 'medium':
        return 'bg-warning-100 text-warning-800 border-warning-200';
      case 'low':
        return 'bg-success-100 text-success-800 border-success-200';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getPriorityIcon = (priority: string) => {
    switch (priority.toLowerCase()) {
      case 'high':
        return <Star className="h-4 w-4" />;
      case 'medium':
        return <TrendingUp className="h-4 w-4" />;
      case 'low':
        return <Clock className="h-4 w-4" />;
      default:
        return <Lightbulb className="h-4 w-4" />;
    }
  };

  const handleCopy = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy text: ', err);
    }
  };

  const handleExpand = () => {
    if (onExpand) {
      onExpand(index);
    }
  };

  return (
    <motion.div
      data-testid="optimization-card"
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="bg-white rounded-lg border border-gray-200 shadow-sm hover:shadow-md transition-shadow"
    >
      {/* Header */}
      <div
        className="p-4 cursor-pointer"
        onClick={handleExpand}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`px-2 py-1 rounded-full text-xs font-medium flex items-center space-x-1 border ${getPriorityColor(suggestion.priority)}`}>
              {getPriorityIcon(suggestion.priority)}
              <span className="capitalize">{suggestion.priority}</span>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 capitalize">
                {suggestion.type.replace(/_/g, ' ')}
              </h3>
              <p className="text-sm text-gray-600">
                {suggestion.description}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            {suggestion.potential_savings && (
              <div className="text-right">
                <div className="text-sm font-medium text-green-600">
                  {suggestion.potential_savings}
                </div>
                <div className="text-xs text-gray-500">Potential Savings</div>
              </div>
            )}
            <motion.div
              animate={{ rotate: isExpanded ? 90 : 0 }}
              transition={{ duration: 0.2 }}
            >
              <ChevronRight className="h-5 w-5 text-gray-400" />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Expanded Content */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            transition={{ duration: 0.3 }}
            className="border-t border-gray-200"
          >
            <div className="p-4 space-y-4">
              {/* Implementation Details */}
              {suggestion.implementation && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Implementation</h4>
                  <div className="p-3 bg-blue-50 border border-blue-200 rounded-md">
                    <p className="text-sm text-blue-800 leading-relaxed">
                      {suggestion.implementation}
                    </p>
                  </div>
                </div>
              )}

              {/* Potential Savings */}
              {suggestion.potential_savings && (
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 mb-2">Potential Savings</h4>
                  <div className="p-3 bg-green-50 border border-green-200 rounded-md">
                    <div className="flex items-center space-x-2">
                      <TrendingUp className="h-4 w-4 text-green-600" />
                      <span className="text-sm font-medium text-green-800">
                        {suggestion.potential_savings}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Copy Implementation */}
              {suggestion.implementation && (
                <div className="flex justify-end">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleCopy(suggestion.implementation!);
                    }}
                    className="flex items-center space-x-1 px-3 py-2 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors"
                  >
                    {copied ? (
                      <>
                        <CheckCircle className="h-4 w-4" />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="h-4 w-4" />
                        <span>Copy Implementation</span>
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Priority Indicator */}
              <div>
                <h4 className="text-sm font-semibold text-gray-900 mb-2">Priority Level</h4>
                <div className="flex items-center space-x-2">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${getPriorityColor(suggestion.priority)}`}>
                    {getPriorityIcon(suggestion.priority)}
                    <span className="ml-1 capitalize">{suggestion.priority} Priority</span>
                  </div>
                  <div className="text-sm text-gray-600">
                    {suggestion.priority.toLowerCase() === 'high' && 'Address immediately'}
                    {suggestion.priority.toLowerCase() === 'medium' && 'Consider for next update'}
                    {suggestion.priority.toLowerCase() === 'low' && 'Optional improvement'}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default OptimizationCard;
