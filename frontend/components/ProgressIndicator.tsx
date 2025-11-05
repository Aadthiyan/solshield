'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, Clock, AlertCircle } from 'lucide-react';
import { ProgressIndicatorProps } from '@/types';

const ProgressIndicator: React.FC<ProgressIndicatorProps> = ({
  progress,
  status,
  estimatedTime,
  isVisible,
}) => {
  const getStatusIcon = () => {
    if (progress === 100) {
      return <CheckCircle className="h-6 w-6 text-green-300" />;
    }
    if (status.toLowerCase().includes('error')) {
      return <AlertCircle className="h-6 w-6 text-red-300" />;
    }
    return <Clock className="h-6 w-6 text-blue-300" />;
  };

  const getStatusColor = () => {
    if (progress === 100) return 'bg-green-500';
    if (status.toLowerCase().includes('error')) return 'bg-red-500';
    return 'bg-blue-500';
  };

  const formatTime = (seconds: number) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.round(seconds % 60);
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (!isVisible) return null;

  return (
    <motion.div
      data-testid="progress-indicator"
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -30 }}
      className="w-full max-w-2xl mx-auto bg-white/10 backdrop-blur-lg rounded-2xl shadow-2xl p-8 border border-white/20"
    >
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
              {getStatusIcon()}
            </div>
            <span className="font-bold text-white text-lg">Analysis Progress</span>
          </div>
          <div className="bg-white/10 backdrop-blur-sm rounded-xl px-4 py-2">
            <span className="text-white font-bold text-lg">{Math.round(progress)}%</span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="progress progress-animated">
          <motion.div
            className={`progress-primary ${progress < 100 ? 'progress-animated' : ''}`}
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.8, ease: 'easeOut' }}
          />
        </div>

        {/* Status Text */}
        <div className="text-center space-y-2">
          <p className="text-white font-semibold text-lg">{status}</p>
          {estimatedTime && estimatedTime > 0 && progress < 100 && (
            <div className="bg-white/10 backdrop-blur-sm rounded-xl px-4 py-2 border border-white/20">
              <p className="text-white/80 text-sm">
                Estimated time remaining: {formatTime(estimatedTime)}
              </p>
            </div>
          )}
        </div>

        {/* Progress Steps */}
        <div className="space-y-3">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${progress >= 20 ? 'bg-green-400 shadow-lg' : 'bg-white/30'}`} />
            <span className={`text-sm font-medium ${progress >= 20 ? 'text-green-300' : 'text-white/60'}`}>
              File Upload
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${progress >= 40 ? 'bg-green-400 shadow-lg' : 'bg-white/30'}`} />
            <span className={`text-sm font-medium ${progress >= 40 ? 'text-green-300' : 'text-white/60'}`}>
              Code Analysis
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${progress >= 70 ? 'bg-green-400 shadow-lg' : 'bg-white/30'}`} />
            <span className={`text-sm font-medium ${progress >= 70 ? 'text-green-300' : 'text-white/60'}`}>
              Vulnerability Detection
            </span>
          </div>
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full ${progress >= 90 ? 'bg-green-400 shadow-lg' : 'bg-white/30'}`} />
            <span className={`text-sm font-medium ${progress >= 90 ? 'text-green-300' : 'text-white/60'}`}>
              Report Generation
            </span>
          </div>
        </div>

        {/* Completion Message */}
        {progress === 100 && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center p-6 bg-green-500/20 backdrop-blur-sm border border-green-400/30 rounded-2xl shadow-2xl"
          >
            <p className="text-lg text-green-300 font-bold">
              Analysis Complete!
            </p>
            <p className="text-sm text-green-200 mt-2">
              Your vulnerability report is ready.
            </p>
          </motion.div>
        )}

        {/* Error Message */}
        {status.toLowerCase().includes('error') && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            className="text-center p-6 bg-red-500/20 backdrop-blur-sm border border-red-400/30 rounded-2xl shadow-2xl"
          >
            <p className="text-lg text-red-300 font-bold">
              Analysis Failed
            </p>
            <p className="text-sm text-red-200 mt-2">
              {status}
            </p>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
};

export default ProgressIndicator;
