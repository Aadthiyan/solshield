'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  BarChart3, 
  Brain,
  Target,
  Zap,
  TrendingUp,
  Eye,
  EyeOff,
  Download,
  Share2,
  ExternalLink,
  ChevronDown,
  ChevronUp,
  Code,
  GitBranch,
  Lock
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';
import GraphVisualization from './GraphVisualization';
import ProxyLabelingInsights from './ProxyLabelingInsights';
import RobustnessDashboard from './RobustnessDashboard';

interface EnhancedVulnerabilityReport {
  report_id: string;
  contract_name: string;
  analysis_timestamp: string;
  risk_score: number;
  is_vulnerable: boolean;
  
  // Enhanced metrics
  enhanced_metrics: {
    overall_confidence: number;
    model_agreement: number;
    proxy_signal_quality: number;
    robustness_score: number;
    joint_gnn_contribution: number;
    codebert_contribution: number;
    gnn_contribution: number;
    fusion_effectiveness: number;
  };
  
  // Joint syntax-semantic analysis
  joint_analysis: {
    syntax_score: number;
    semantic_score: number;
    interaction_score: number;
    graph_complexity: number;
    node_count: number;
    edge_count: number;
  };
  
  // Proxy labeling insights
  proxy_labels: {
    explicit_label: number;
    proxy_signals: any[];
    proxy_scores: any;
    soft_labels: any;
    augmented_labels: any;
    confidence: number;
  };
  
  // Robustness metrics
  robustness_metrics: {
    overall_robustness: number;
    adversarial_accuracy: number;
    detection_rate: number;
    false_positive_rate: number;
    defense_effectiveness: number;
    model_stability: number;
  };
  
  // Traditional vulnerability data
  vulnerabilities: Array<{
    id: string;
    type: string;
    severity: 'critical' | 'high' | 'medium' | 'low';
    description: string;
    line_number: number;
    confidence: number;
    recommendation: string;
    joint_analysis_confidence: number;
    proxy_signal_support: number;
    robustness_confidence: number;
  }>;
  
  optimization_suggestions: Array<{
    id: string;
    type: string;
    description: string;
    potential_savings: string;
    line_number: number;
    confidence: number;
  }>;
  
  summary: {
    total_vulnerabilities: number;
    critical_vulnerabilities: number;
    high_vulnerabilities: number;
    medium_vulnerabilities: number;
    low_vulnerabilities: number;
  };
}

interface EnhancedReportDisplayProps {
  report: EnhancedVulnerabilityReport;
  onClose?: () => void;
  showDetails?: boolean;
}

const EnhancedReportDisplay: React.FC<EnhancedReportDisplayProps> = ({
  report,
  onClose,
  showDetails = true
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'graph' | 'proxy' | 'robustness' | 'details'>('overview');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['summary']));
  const [showAdvancedMetrics, setShowAdvancedMetrics] = useState(false);

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#ef4444';
      case 'high': return '#f59e0b';
      case 'medium': return '#3b82f6';
      case 'low': return '#10b981';
      default: return '#6b7280';
    }
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'critical': return AlertTriangle;
      case 'high': return AlertTriangle;
      case 'medium': return Info;
      case 'low': return CheckCircle;
      default: return Info;
    }
  };

  // Prepare data for charts
  const vulnerabilityDistribution = [
    { name: 'Critical', value: report.summary.critical_vulnerabilities, color: '#ef4444' },
    { name: 'High', value: report.summary.high_vulnerabilities, color: '#f59e0b' },
    { name: 'Medium', value: report.summary.medium_vulnerabilities, color: '#3b82f6' },
    { name: 'Low', value: report.summary.low_vulnerabilities, color: '#10b981' }
  ];

  const modelContributionData = [
    { name: 'Joint GNN', value: report.enhanced_metrics.joint_gnn_contribution, color: '#8b5cf6' },
    { name: 'CodeBERT', value: report.enhanced_metrics.codebert_contribution, color: '#3b82f6' },
    { name: 'GNN', value: report.enhanced_metrics.gnn_contribution, color: '#06b6d4' },
    { name: 'Fusion', value: report.enhanced_metrics.fusion_effectiveness, color: '#10b981' }
  ];

  return (
    <div className="w-full max-w-7xl mx-auto bg-white/5 backdrop-blur-sm rounded-2xl border border-white/20 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-white/20">
        <div className="flex items-center space-x-4">
          <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl">
            <Shield className="h-8 w-8 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">Enhanced Vulnerability Report</h2>
            <p className="text-sm text-white/70">{report.contract_name} • {report.analysis_timestamp}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowAdvancedMetrics(!showAdvancedMetrics)}
            className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300"
          >
            {showAdvancedMetrics ? <EyeOff className="h-5 w-5 text-white" /> : <Eye className="h-5 w-5 text-white" />}
            <span className="ml-2 text-sm text-white">Advanced</span>
          </button>
          <button className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300">
            <Download className="h-5 w-5 text-white" />
          </button>
          <button className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300">
            <Share2 className="h-5 w-5 text-white" />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300"
            >
              <ExternalLink className="h-5 w-5 text-white" />
            </button>
          )}
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center space-x-1 p-4 border-b border-white/20">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'graph', label: 'Joint Graph', icon: GitBranch },
          { id: 'proxy', label: 'Proxy Labels', icon: Brain },
          { id: 'robustness', label: 'Robustness', icon: Shield },
          { id: 'details', label: 'Details', icon: Info }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
              activeTab === tab.id
                ? 'bg-white/20 text-white shadow-lg'
                : 'text-white/70 hover:text-white hover:bg-white/10'
            }`}
          >
            <tab.icon className="h-4 w-4" />
            <span className="text-sm font-medium">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeTab === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Risk Score and Status */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gradient-to-r from-red-500/20 to-orange-500/20 rounded-xl p-4 border border-red-400/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white/70">Risk Score</p>
                      <p className="text-3xl font-bold text-white">{report.risk_score}/10</p>
                    </div>
                    <AlertTriangle className="h-8 w-8 text-red-300" />
                  </div>
                  <div className="mt-2">
                    <div className="w-full bg-white/20 rounded-full h-3">
                      <div 
                        className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full transition-all duration-1000"
                        style={{ width: `${report.risk_score * 10}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl p-4 border border-blue-400/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white/70">Overall Confidence</p>
                      <p className="text-3xl font-bold text-white">
                        {Math.round(report.enhanced_metrics.overall_confidence)}%
                      </p>
                    </div>
                    <Target className="h-8 w-8 text-blue-300" />
                  </div>
                  <div className="mt-2">
                    <div className="w-full bg-white/20 rounded-full h-3">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-1000"
                        style={{ width: `${report.enhanced_metrics.overall_confidence}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-green-500/20 to-cyan-500/20 rounded-xl p-4 border border-green-400/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white/70">Robustness Score</p>
                      <p className="text-3xl font-bold text-white">
                        {Math.round(report.robustness_metrics.overall_robustness)}%
                      </p>
                    </div>
                    <Shield className="h-8 w-8 text-green-300" />
                  </div>
                  <div className="mt-2">
                    <div className="w-full bg-white/20 rounded-full h-3">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-cyan-500 h-3 rounded-full transition-all duration-1000"
                        style={{ width: `${report.robustness_metrics.overall_robustness}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Vulnerability Distribution */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Vulnerability Distribution</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={vulnerabilityDistribution}
                          cx="50%"
                          cy="50%"
                          innerRadius={60}
                          outerRadius={100}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {vulnerabilityDistribution.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip 
                          contentStyle={{
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            borderRadius: '8px',
                            color: 'white'
                          }}
                        />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Model Contribution</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={modelContributionData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                        <XAxis dataKey="name" stroke="rgba(255, 255, 255, 0.7)" />
                        <YAxis stroke="rgba(255, 255, 255, 0.7)" />
                        <Tooltip 
                          contentStyle={{
                            backgroundColor: 'rgba(0, 0, 0, 0.8)',
                            border: '1px solid rgba(255, 255, 255, 0.2)',
                            borderRadius: '8px',
                            color: 'white'
                          }}
                        />
                        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                          {modelContributionData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Advanced Metrics (if enabled) */}
              {showAdvancedMetrics && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="bg-white/5 rounded-xl p-4 border border-white/10"
                >
                  <h4 className="text-lg font-semibold text-white mb-4">Advanced Metrics</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <p className="text-sm text-white/70">Model Agreement</p>
                      <p className="text-xl font-bold text-white">
                        {Math.round(report.enhanced_metrics.model_agreement)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-white/70">Proxy Signal Quality</p>
                      <p className="text-xl font-bold text-white">
                        {Math.round(report.enhanced_metrics.proxy_signal_quality)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-white/70">Fusion Effectiveness</p>
                      <p className="text-xl font-bold text-white">
                        {Math.round(report.enhanced_metrics.fusion_effectiveness)}%
                      </p>
                    </div>
                    <div className="text-center">
                      <p className="text-sm text-white/70">Model Stability</p>
                      <p className="text-xl font-bold text-white">
                        {Math.round(report.robustness_metrics.model_stability)}%
                      </p>
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Vulnerabilities List */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-white">Detected Vulnerabilities</h4>
                  <span className="text-sm text-white/70">
                    {report.summary.total_vulnerabilities} vulnerabilities found
                  </span>
                </div>
                
                <div className="space-y-3">
                  {report.vulnerabilities.map((vuln, index) => {
                    const SeverityIcon = getSeverityIcon(vuln.severity);
                    return (
                      <motion.div
                        key={vuln.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.1 }}
                        className="bg-white/5 rounded-lg p-4 border border-white/10 hover:bg-white/10 transition-all duration-300"
                      >
                        <div className="flex items-start justify-between">
                          <div className="flex items-start space-x-3">
                            <div 
                              className="p-2 rounded-lg"
                              style={{ backgroundColor: getSeverityColor(vuln.severity) + '20' }}
                            >
                              <SeverityIcon 
                                className="h-5 w-5" 
                                style={{ color: getSeverityColor(vuln.severity) }}
                              />
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-2">
                                <h5 className="font-semibold text-white">{vuln.type}</h5>
                                <span 
                                  className="px-2 py-1 rounded-full text-xs font-medium"
                                  style={{ 
                                    backgroundColor: getSeverityColor(vuln.severity) + '20',
                                    color: getSeverityColor(vuln.severity)
                                  }}
                                >
                                  {vuln.severity.toUpperCase()}
                                </span>
                                <span className="text-xs text-white/60">Line {vuln.line_number}</span>
                              </div>
                              <p className="text-sm text-white/70 mb-2">{vuln.description}</p>
                              <p className="text-sm text-green-300 mb-2">{vuln.recommendation}</p>
                              
                              {/* Enhanced confidence indicators */}
                              <div className="flex items-center space-x-4 text-xs">
                                <div className="flex items-center space-x-1">
                                  <span className="text-white/60">Joint Analysis:</span>
                                  <span className="text-white font-medium">
                                    {Math.round(vuln.joint_analysis_confidence)}%
                                  </span>
                                </div>
                                <div className="flex items-center space-x-1">
                                  <span className="text-white/60">Proxy Support:</span>
                                  <span className="text-white font-medium">
                                    {Math.round(vuln.proxy_signal_support)}%
                                  </span>
                                </div>
                                <div className="flex items-center space-x-1">
                                  <span className="text-white/60">Robustness:</span>
                                  <span className="text-white font-medium">
                                    {Math.round(vuln.robustness_confidence)}%
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-lg font-bold text-white">
                              {Math.round(vuln.confidence * 100)}%
                            </div>
                            <div className="text-sm text-white/70">Confidence</div>
                          </div>
                        </div>
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'graph' && (
            <motion.div
              key="graph"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <GraphVisualization
                nodes={[]} // Would be populated with actual graph data
                edges={[]} // Would be populated with actual graph data
                contractName={report.contract_name}
                showVulnerabilities={true}
                showSecurityPatterns={true}
                showSemanticFlow={true}
              />
            </motion.div>
          )}

          {activeTab === 'proxy' && (
            <motion.div
              key="proxy"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <ProxyLabelingInsights
                proxyLabels={report.proxy_labels}
              />
            </motion.div>
          )}

          {activeTab === 'robustness' && (
            <motion.div
              key="robustness"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <RobustnessDashboard
                robustnessMetrics={report.robustness_metrics}
                adversarialTests={[]} // Would be populated with actual test data
                defenseStatus="active"
              />
            </motion.div>
          )}

          {activeTab === 'details' && (
            <motion.div
              key="details"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Joint Analysis Details */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <button
                  onClick={() => toggleSection('joint-analysis')}
                  className="flex items-center justify-between w-full text-left"
                >
                  <h4 className="text-lg font-semibold text-white">Joint Syntax-Semantic Analysis</h4>
                  {expandedSections.has('joint-analysis') ? 
                    <ChevronUp className="h-5 w-5 text-white" /> : 
                    <ChevronDown className="h-5 w-5 text-white" />
                  }
                </button>
                
                <AnimatePresence>
                  {expandedSections.has('joint-analysis') && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 space-y-4"
                    >
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="text-center">
                          <p className="text-sm text-white/70">Syntax Score</p>
                          <p className="text-xl font-bold text-white">
                            {Math.round(report.joint_analysis.syntax_score)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-white/70">Semantic Score</p>
                          <p className="text-xl font-bold text-white">
                            {Math.round(report.joint_analysis.semantic_score)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-white/70">Interaction Score</p>
                          <p className="text-xl font-bold text-white">
                            {Math.round(report.joint_analysis.interaction_score)}%
                          </p>
                        </div>
                        <div className="text-center">
                          <p className="text-sm text-white/70">Graph Complexity</p>
                          <p className="text-xl font-bold text-white">
                            {Math.round(report.joint_analysis.graph_complexity)}%
                          </p>
                        </div>
                      </div>
                      <div className="text-sm text-white/70">
                        Graph contains {report.joint_analysis.node_count} nodes and {report.joint_analysis.edge_count} edges
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>

              {/* Optimization Suggestions */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <button
                  onClick={() => toggleSection('optimization')}
                  className="flex items-center justify-between w-full text-left"
                >
                  <h4 className="text-lg font-semibold text-white">Optimization Suggestions</h4>
                  {expandedSections.has('optimization') ? 
                    <ChevronUp className="h-5 w-5 text-white" /> : 
                    <ChevronDown className="h-5 w-5 text-white" />
                  }
                </button>
                
                <AnimatePresence>
                  {expandedSections.has('optimization') && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 space-y-3"
                    >
                      {report.optimization_suggestions.map((suggestion, index) => (
                        <div key={suggestion.id} className="bg-white/5 rounded-lg p-3 border border-white/10">
                          <div className="flex items-start justify-between">
                            <div className="flex-1">
                              <div className="flex items-center space-x-2 mb-1">
                                <h5 className="font-semibold text-white">{suggestion.type}</h5>
                                <span className="text-xs text-white/60">Line {suggestion.line_number}</span>
                              </div>
                              <p className="text-sm text-white/70 mb-2">{suggestion.description}</p>
                              <p className="text-sm text-green-300">Potential savings: {suggestion.potential_savings}</p>
                            </div>
                            <div className="text-right">
                              <div className="text-sm font-bold text-white">
                                {Math.round(suggestion.confidence * 100)}%
                              </div>
                              <div className="text-xs text-white/70">Confidence</div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default EnhancedReportDisplay;
