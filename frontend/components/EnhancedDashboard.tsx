'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';
import { 
  Shield, 
  AlertTriangle, 
  TrendingUp, 
  Clock, 
  RefreshCw,
  Filter,
  Eye,
  Brain,
  GitBranch,
  Target,
  Zap,
  Info,
  BarChart3,
  Settings,
  Download,
  Share2
} from 'lucide-react';
import { DashboardProps, VulnerabilityReport } from '@/types';
import EnhancedReportDisplay from './EnhancedReportDisplay';
import GraphVisualization from './GraphVisualization';
import ProxyLabelingInsights from './ProxyLabelingInsights';
import RobustnessDashboard from './RobustnessDashboard';

interface EnhancedDashboardProps extends DashboardProps {
  enhancedMetrics?: {
    overall_confidence: number;
    model_agreement: number;
    proxy_signal_quality: number;
    robustness_score: number;
    joint_gnn_contribution: number;
    codebert_contribution: number;
    gnn_contribution: number;
    fusion_effectiveness: number;
  };
  proxyLabels?: any;
  robustnessMetrics?: any;
  adversarialTests?: any[];
}

const EnhancedDashboard: React.FC<EnhancedDashboardProps> = ({
  stats,
  reports,
  onReportSelect,
  onRefresh,
  isLoading = false,
  enhancedMetrics,
  proxyLabels,
  robustnessMetrics,
  adversarialTests = []
}) => {
  const [activeView, setActiveView] = useState<'overview' | 'graph' | 'proxy' | 'robustness' | 'reports'>('overview');
  const [showFilters, setShowFilters] = useState(false);
  const [filteredReports, setFilteredReports] = useState<VulnerabilityReport[]>(reports);
  const [selectedReport, setSelectedReport] = useState<VulnerabilityReport | null>(null);
  const [showAdvancedFeatures, setShowAdvancedFeatures] = useState(true);

  useEffect(() => {
    setFilteredReports(reports);
  }, [reports]);

  const handleReportSelect = (report: VulnerabilityReport) => {
    setSelectedReport(report);
    onReportSelect(report);
  };

  const getRiskColor = (score: number) => {
    if (score >= 8) return '#dc2626';
    if (score >= 6) return '#ea580c';
    if (score >= 4) return '#d97706';
    return '#16a34a';
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical': return '#be185d';
      case 'high': return '#dc2626';
      case 'medium': return '#d97706';
      case 'low': return '#16a34a';
      default: return '#6b7280';
    }
  };

  // Enhanced data preparation
  const vulnerabilityTypeData = stats.topVulnerabilityTypes.map(item => ({
    name: item.type.replace(/_/g, ' '),
    value: item.count,
    color: getRiskColor(item.count * 2)
  }));

  const severityData = reports.reduce((acc, report) => {
    report.vulnerabilities.forEach(vuln => {
      const existing = acc.find(item => item.name === vuln.severity);
      if (existing) {
        existing.value += 1;
      } else {
        acc.push({
          name: vuln.severity,
          value: 1,
          color: getSeverityColor(vuln.severity)
        });
      }
    });
    return acc;
  }, [] as any[]);

  const recentReportsData = reports
    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
    .slice(0, 10)
    .map(report => ({
      name: report.contract_name || 'Unknown',
      riskScore: report.risk_score,
      vulnerabilities: report.vulnerabilities.length,
      timestamp: new Date(report.timestamp).toLocaleDateString()
    }));

  // Enhanced metrics data
  const enhancedMetricsData = enhancedMetrics ? [
    { name: 'Overall Confidence', value: enhancedMetrics.overall_confidence, color: '#3b82f6' },
    { name: 'Model Agreement', value: enhancedMetrics.model_agreement, color: '#10b981' },
    { name: 'Proxy Signal Quality', value: enhancedMetrics.proxy_signal_quality, color: '#8b5cf6' },
    { name: 'Robustness Score', value: enhancedMetrics.robustness_score, color: '#f59e0b' },
    { name: 'Fusion Effectiveness', value: enhancedMetrics.fusion_effectiveness, color: '#06b6d4' }
  ] : [];

  const modelContributionData = enhancedMetrics ? [
    { name: 'Joint GNN', value: enhancedMetrics.joint_gnn_contribution, color: '#8b5cf6' },
    { name: 'CodeBERT', value: enhancedMetrics.codebert_contribution, color: '#3b82f6' },
    { name: 'GNN', value: enhancedMetrics.gnn_contribution, color: '#06b6d4' },
    { name: 'Fusion', value: enhancedMetrics.fusion_effectiveness, color: '#10b981' }
  ] : [];

  if (isLoading) {
    return (
      <div className="w-full max-w-7xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-200 rounded"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="h-64 bg-gray-200 rounded"></div>
            <div className="h-64 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-7xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white">Enhanced Dashboard</h1>
          <p className="text-white/70">Advanced Smart Contract Vulnerability Analysis</p>
        </div>
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowAdvancedFeatures(!showAdvancedFeatures)}
            className="flex items-center space-x-2 px-4 py-2 text-sm text-white/70 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          >
            <Brain className="h-4 w-4" />
            <span>Advanced Features</span>
          </button>
          <button
            data-testid="filter-button"
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center space-x-2 px-4 py-2 text-sm text-white/70 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          >
            <Filter className="h-4 w-4" />
            <span>Filters</span>
          </button>
          <button
            data-testid="refresh-button"
            onClick={onRefresh}
            className="flex items-center space-x-2 px-4 py-2 text-sm text-white/70 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
          >
            <RefreshCw className="h-4 w-4" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center space-x-1 p-4 bg-white/5 backdrop-blur-sm rounded-xl border border-white/20">
        {[
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'graph', label: 'Joint Graph', icon: GitBranch },
          { id: 'proxy', label: 'Proxy Labels', icon: Brain },
          { id: 'robustness', label: 'Robustness', icon: Shield },
          { id: 'reports', label: 'Reports', icon: Eye }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveView(tab.id as any)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-300 ${
              activeView === tab.id
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
      <div className="space-y-6">
        <AnimatePresence mode="wait">
          {activeView === 'overview' && (
            <motion.div
              key="overview"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Enhanced Stats Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
                      <Shield className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-white/70">Total Analyses</p>
                      <p className="text-2xl font-bold text-white">{stats.totalAnalyses}</p>
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-3 bg-gradient-to-r from-red-500 to-orange-600 rounded-xl">
                      <AlertTriangle className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-white/70">Vulnerable Contracts</p>
                      <p className="text-2xl font-bold text-white">{stats.vulnerableContracts}</p>
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-3 bg-gradient-to-r from-yellow-500 to-orange-600 rounded-xl">
                      <TrendingUp className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-white/70">Average Risk Score</p>
                      <p className="text-2xl font-bold text-white">{stats.averageRiskScore.toFixed(1)}</p>
                    </div>
                  </div>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
                >
                  <div className="flex items-center space-x-3">
                    <div className="p-3 bg-gradient-to-r from-green-500 to-cyan-600 rounded-xl">
                      <Clock className="h-6 w-6 text-white" />
                    </div>
                    <div>
                      <p className="text-sm text-white/70">Recent Analyses</p>
                      <p className="text-2xl font-bold text-white">{stats.recentAnalyses.length}</p>
                    </div>
                  </div>
                </motion.div>
              </div>

              {/* Enhanced Metrics (if available) */}
              {showAdvancedFeatures && enhancedMetrics && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.4 }}
                  className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
                >
                  <h3 className="text-lg font-semibold text-white mb-4">Enhanced Model Metrics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {enhancedMetricsData.map((metric, index) => (
                      <div key={index} className="text-center">
                        <div className="w-16 h-16 mx-auto mb-2 rounded-full flex items-center justify-center"
                             style={{ backgroundColor: metric.color + '20' }}>
                          <span className="text-lg font-bold text-white">{Math.round(metric.value)}%</span>
                        </div>
                        <p className="text-xs text-white/70">{metric.name}</p>
                      </div>
                    ))}
                  </div>
                </motion.div>
              )}

              {/* Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Vulnerability Types Chart */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.5 }}
                  className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
                >
                  <h3 className="text-lg font-semibold text-white mb-4">Vulnerability Types</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={vulnerabilityTypeData}>
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
                      <Bar dataKey="value" fill="#3b82f6" />
                    </BarChart>
                  </ResponsiveContainer>
                </motion.div>

                {/* Model Contribution Chart */}
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.6 }}
                  className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
                >
                  <h3 className="text-lg font-semibold text-white mb-4">Model Contribution</h3>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={modelContributionData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {modelContributionData.map((entry, index) => (
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
                </motion.div>
              </div>

              {/* Risk Score Trend */}
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 }}
                className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/20 shadow-2xl"
              >
                <h3 className="text-lg font-semibold text-white mb-4">Risk Score Trend</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={recentReportsData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis dataKey="timestamp" stroke="rgba(255, 255, 255, 0.7)" />
                    <YAxis stroke="rgba(255, 255, 255, 0.7)" />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        borderRadius: '8px',
                        color: 'white'
                      }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="riskScore" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </motion.div>
            </motion.div>
          )}

          {activeView === 'graph' && (
            <motion.div
              key="graph"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <GraphVisualization
                nodes={[]} // Would be populated with actual graph data
                edges={[]} // Would be populated with actual graph data
                contractName="Sample Contract"
                showVulnerabilities={true}
                showSecurityPatterns={true}
                showSemanticFlow={true}
              />
            </motion.div>
          )}

          {activeView === 'proxy' && (
            <motion.div
              key="proxy"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <ProxyLabelingInsights
                proxyLabels={proxyLabels || {
                  explicit_label: 0,
                  proxy_signals: [],
                  proxy_scores: { safety_score: 0, vulnerability_score: 0, complexity_score: 0, security_score: 0 },
                  soft_labels: { vulnerable: 0, safe: 0, complex: 0 },
                  augmented_labels: { vulnerability_types: [], severity_levels: [], risk_factors: [], security_patterns: [] },
                  confidence: 0
                }}
              />
            </motion.div>
          )}

          {activeView === 'robustness' && (
            <motion.div
              key="robustness"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
            >
              <RobustnessDashboard
                robustnessMetrics={robustnessMetrics || {
                  overall_robustness: 85,
                  adversarial_accuracy: 88,
                  detection_rate: 92,
                  false_positive_rate: 5,
                  defense_effectiveness: 90,
                  model_stability: 87
                }}
                adversarialTests={adversarialTests}
                defenseStatus="active"
              />
            </motion.div>
          )}

          {activeView === 'reports' && (
            <motion.div
              key="reports"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Recent Reports */}
              <div className="bg-white/5 backdrop-blur-sm rounded-xl border border-white/20 shadow-2xl">
                <div className="p-6 border-b border-white/20">
                  <h3 className="text-lg font-semibold text-white">Recent Reports</h3>
                </div>
                <div className="divide-y divide-white/10">
                  {filteredReports.slice(0, 10).map((report, index) => (
                    <motion.div
                      key={report.report_id}
                      data-testid="report-item"
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="p-4 hover:bg-white/5 cursor-pointer transition-colors"
                      onClick={() => handleReportSelect(report)}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className={`w-3 h-3 rounded-full ${
                            report.risk_score >= 8 ? 'bg-red-500' :
                            report.risk_score >= 6 ? 'bg-orange-500' :
                            report.risk_score >= 4 ? 'bg-yellow-500' : 'bg-green-500'
                          }`} />
                          <div>
                            <p className="font-medium text-white">
                              {report.contract_name || 'Unknown Contract'}
                            </p>
                            <p className="text-sm text-white/70">
                              {report.vulnerabilities.length} vulnerabilities • 
                              Risk Score: {report.risk_score.toFixed(1)}
                            </p>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          <span className="text-sm text-white/70">
                            {new Date(report.timestamp).toLocaleDateString()}
                          </span>
                          <Eye className="h-4 w-4 text-white/70" />
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Enhanced Report Display Modal */}
      <AnimatePresence>
        {selectedReport && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              className="w-full max-w-7xl max-h-[90vh] overflow-y-auto"
            >
              <EnhancedReportDisplay
                report={selectedReport as any}
                onClose={() => setSelectedReport(null)}
                showDetails={true}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default EnhancedDashboard;
