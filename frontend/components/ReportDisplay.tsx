'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Copy,
  X
} from 'lucide-react';
import { ReportDisplayProps } from '@/types';
import VulnerabilityCard from './VulnerabilityCard';
import OptimizationCard from './OptimizationCard';

const ReportDisplay: React.FC<ReportDisplayProps> = ({
  report,
  isLoading = false,
  onClose,
}) => {
  const [expandedVulnerabilities, setExpandedVulnerabilities] = useState<Set<number>>(new Set());
  const [expandedOptimizations, setExpandedOptimizations] = useState<Set<number>>(new Set());
  const [copied, setCopied] = useState(false);

  const handleVulnerabilityExpand = (index: number) => {
    const newExpanded = new Set(expandedVulnerabilities);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedVulnerabilities(newExpanded);
  };

  const handleOptimizationExpand = (index: number) => {
    const newExpanded = new Set(expandedOptimizations);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedOptimizations(newExpanded);
  };

  const handleCopyReport = async () => {
    try {
      const reportText = `Smart Contract Vulnerability Report
Report ID: ${report.report_id}
Contract: ${report.contract_name || 'Unknown'}
Risk Score: ${report.risk_score}/10
Vulnerabilities: ${report.vulnerabilities.length}
Timestamp: ${new Date(report.timestamp).toLocaleString()}

${report.vulnerabilities.map((vuln, index) => 
  `${index + 1}. ${vuln.type} (${vuln.severity}) - ${vuln.description}`
).join('\n')}`;

      await navigator.clipboard.writeText(reportText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy report: ', err);
    }
  };

  const getRiskColor = (score: number) => {
    if (score >= 8) return 'text-critical-600';
    if (score >= 6) return 'text-error-600';
    if (score >= 4) return 'text-warning-600';
    return 'text-success-600';
  };

  const getRiskBackground = (score: number) => {
    if (score >= 8) return 'bg-critical-50 border-critical-200';
    if (score >= 6) return 'bg-error-50 border-error-200';
    if (score >= 4) return 'bg-warning-50 border-warning-200';
    return 'bg-success-50 border-success-200';
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  if (isLoading) {
    return (
      <div className="w-full max-w-4xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded"></div>
          <div className="h-32 bg-gray-200 rounded"></div>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-24 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full max-w-4xl mx-auto bg-white rounded-lg shadow-lg"
    >
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-primary-100 rounded-lg">
              <Shield className="h-6 w-6 text-primary-600" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Vulnerability Report
              </h1>
              <p className="text-sm text-gray-600">
                {report.contract_name || 'Smart Contract'} • {formatDate(report.timestamp)}
              </p>
            </div>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              data-testid="copy-report-button"
              onClick={handleCopyReport}
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
                  <span>Copy</span>
                </>
              )}
            </button>
            
            {onClose && (
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-md transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="p-6 border-b border-gray-200">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Risk Score */}
          <div className={`p-4 rounded-lg border-2 ${getRiskBackground(report.risk_score)}`}>
            <div className="flex items-center space-x-2">
              <AlertTriangle className={`h-5 w-5 ${getRiskColor(report.risk_score)}`} />
              <span className="font-semibold text-gray-900">Risk Score</span>
            </div>
            <div className={`text-3xl font-bold mt-2 ${getRiskColor(report.risk_score)}`}>
              {report.risk_score.toFixed(1)}
            </div>
            <div className="text-sm text-gray-600">out of 10</div>
          </div>

          {/* Vulnerabilities */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <Shield className="h-5 w-5 text-gray-600" />
              <span className="font-semibold text-gray-900">Vulnerabilities</span>
            </div>
            <div className="text-3xl font-bold text-gray-900 mt-2">
              {report.vulnerabilities.length}
            </div>
            <div className="text-sm text-gray-600">
              {report.is_vulnerable ? 'Issues found' : 'No issues'}
            </div>
          </div>

          {/* Confidence */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-5 w-5 text-gray-600" />
              <span className="font-semibold text-gray-900">Confidence</span>
            </div>
            <div className="text-3xl font-bold text-gray-900 mt-2">
              {Math.round(report.overall_confidence * 100)}%
            </div>
            <div className="text-sm text-gray-600">Analysis confidence</div>
          </div>
        </div>
      </div>

      {/* Vulnerabilities Section */}
      {report.vulnerabilities.length > 0 && (
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">
              Vulnerabilities ({report.vulnerabilities.length})
            </h2>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  const allExpanded = report.vulnerabilities.length === expandedVulnerabilities.size;
                  if (allExpanded) {
                    setExpandedVulnerabilities(new Set());
                  } else {
                    setExpandedVulnerabilities(new Set(report.vulnerabilities.map((_, i) => i)));
                  }
                }}
                className="text-sm text-primary-600 hover:text-primary-800"
              >
                {expandedVulnerabilities.size === report.vulnerabilities.length ? 'Collapse All' : 'Expand All'}
              </button>
            </div>
          </div>
          
          <div className="space-y-4">
            {report.vulnerabilities.map((vulnerability, index) => (
              <VulnerabilityCard
                key={index}
                vulnerability={vulnerability}
                index={index}
                onExpand={handleVulnerabilityExpand}
                isExpanded={expandedVulnerabilities.has(index)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Optimization Suggestions */}
      {report.optimization_suggestions.length > 0 && (
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">
              Optimization Suggestions ({report.optimization_suggestions.length})
            </h2>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => {
                  const allExpanded = report.optimization_suggestions.length === expandedOptimizations.size;
                  if (allExpanded) {
                    setExpandedOptimizations(new Set());
                  } else {
                    setExpandedOptimizations(new Set(report.optimization_suggestions.map((_, i) => i)));
                  }
                }}
                className="text-sm text-primary-600 hover:text-primary-800"
              >
                {expandedOptimizations.size === report.optimization_suggestions.length ? 'Collapse All' : 'Expand All'}
              </button>
            </div>
          </div>
          
          <div className="space-y-4">
            {report.optimization_suggestions.map((suggestion, index) => (
              <OptimizationCard
                key={index}
                suggestion={suggestion}
                index={index}
                onExpand={handleOptimizationExpand}
                isExpanded={expandedOptimizations.has(index)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Model Predictions */}
      {report.model_predictions.length > 0 && (
        <div className="p-6 border-b border-gray-200">
          <h2 className="text-xl font-semibold text-gray-900 mb-4">
            Model Predictions
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {report.model_predictions.map((prediction, index) => (
              <div key={index} className="p-4 bg-gray-50 rounded-lg">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-medium text-gray-900 capitalize">
                    {prediction.model_type}
                  </span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    prediction.is_vulnerable 
                      ? 'bg-red-100 text-red-800' 
                      : 'bg-green-100 text-green-800'
                  }`}>
                    {prediction.is_vulnerable ? 'Vulnerable' : 'Safe'}
                  </span>
                </div>
                <div className="text-sm text-gray-600">
                  Confidence: {Math.round(prediction.confidence * 100)}%
                </div>
                <div className="text-sm text-gray-600">
                  Processing: {prediction.processing_time.toFixed(2)}s
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Footer */}
      <div className="p-6 bg-gray-50 rounded-b-lg">
        <div className="flex items-center justify-between text-sm text-gray-600">
          <div>
            Report ID: {report.report_id}
          </div>
          <div>
            Processing time: {report.processing_time.toFixed(2)}s
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default ReportDisplay;
