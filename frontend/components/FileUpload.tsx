'use client';

import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, FileText, AlertCircle, X } from 'lucide-react';
import { FileUploadProps } from '@/types';

const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  onAnalysisStart,
  isAnalyzing,
  acceptedTypes = ['.sol', '.txt'],
  maxSize = 10 * 1024 * 1024, // 10MB
}) => {
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const onDrop = useCallback((acceptedFiles: File[], rejectedFiles: any[]) => {
    setError(null);
    
    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0];
      if (rejection.errors[0]?.code === 'file-too-large') {
        setError(`File is too large. Maximum size is ${maxSize / (1024 * 1024)}MB`);
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        setError(`Invalid file type. Accepted types: ${acceptedTypes.join(', ')}`);
      } else {
        setError('File upload failed. Please try again.');
      }
      return;
    }

    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setSelectedFile(file);
      onFileSelect(file);
    }
  }, [acceptedTypes, maxSize, onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'text/plain': acceptedTypes,
      'application/sol': ['.sol'],
    },
    maxSize,
    multiple: false,
    disabled: isAnalyzing,
  });

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setError(null);
  };

  const handleAnalyze = () => {
    if (selectedFile) {
      const reader = new FileReader();
      reader.onload = (e) => {
        const contractCode = e.target?.result as string;
        onAnalysisStart({
          contract_code: contractCode,
          contract_name: selectedFile.name.replace(/\.[^/.]+$/, ''),
          model_type: 'ensemble',
          include_optimization_suggestions: true,
          include_explanation: true,
        });
      };
      reader.readAsText(selectedFile);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="w-full max-w-4xl mx-auto">
      <div
        {...getRootProps()}
        data-testid="upload-interface"
        className={`
          relative border-2 border-dashed rounded-xl p-12 text-center cursor-pointer transition-all duration-300
          ${isDragActive || dragActive
            ? 'border-blue-500 bg-blue-50'
            : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }
          ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : ''}
          bg-white
        `}
        onMouseEnter={() => setDragActive(true)}
        onMouseLeave={() => setDragActive(false)}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence>
          {!selectedFile ? (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              className="space-y-6"
            >
              <motion.div 
                className="flex justify-center"
                whileHover={{ scale: 1.1, rotate: 5 }}
                transition={{ type: "spring", stiffness: 300 }}
              >
                <div className="w-20 h-20 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                  <Upload className="h-10 w-10 text-blue-600" />
                </div>
              </motion.div>
              
              <div>
                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {isDragActive ? 'Drop your contract here' : 'Upload Smart Contract'}
                </h3>
                <p className="text-gray-600">
                  Drag and drop your Solidity contract file, or click to browse
                </p>
              </div>
              
              <div className="text-sm text-gray-500 bg-gray-50 rounded-lg px-4 py-2">
                <p>Accepted formats: {acceptedTypes.join(', ')}</p>
                <p>Maximum size: {formatFileSize(maxSize)}</p>
              </div>
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -30 }}
              className="space-y-6"
            >
                <motion.div
                className="flex items-center justify-center space-x-4 bg-blue-50 rounded-lg p-4 border border-blue-200"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2 }}
              >
                <div className="p-2 bg-blue-100 rounded-lg">
                  <FileText className="h-6 w-6 text-blue-600" />
                </div>
                <div className="text-left flex-1">
                  <p className="font-semibold text-gray-900">{selectedFile.name}</p>
                  <p className="text-sm text-gray-600">{formatFileSize(selectedFile.size)}</p>
                </div>
                <button
                  data-testid="remove-file-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleRemoveFile();
                  }}
                  className="p-1 hover:bg-gray-200 rounded transition-colors"
                >
                  <X className="h-5 w-5 text-gray-500 hover:text-gray-700" />
                </button>
              </motion.div>
              
              <motion.div 
                className="flex justify-center"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                <button
                  data-testid="analyze-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleAnalyze();
                  }}
                  disabled={isAnalyzing}
                  className="btn-primary btn-lg btn-glow"
                >
                  {isAnalyzing ? 'Analyzing...' : 'Analyze Contract'}
                </button>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {error && (
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg"
          >
            <div className="flex items-center space-x-3">
              <AlertCircle className="h-5 w-5 text-red-600" />
              <span className="text-red-800 font-semibold">{error}</span>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {isAnalyzing && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg"
        >
          <div className="flex items-center justify-center space-x-3">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
            <span className="text-blue-800 font-medium">Analyzing contract...</span>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default FileUpload;
