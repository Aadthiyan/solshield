'use client';

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  CheckCircle, 
  AlertTriangle, 
  Info, 
  TrendingUp, 
  Target,
  Zap,
  Brain,
  BarChart3,
  Eye,
  EyeOff,
  Filter
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface ProxySignal {
  signal_type: string;
  confidence: float;
  source_pattern: string;
  context: {
    description: string;
    vulnerability_types: string[];
    safety_score?: number;
    vulnerability_score?: number;
  };
  line_number: number;
  importance_weight: number;
}

interface ProxyLabelingInsightsProps {
  proxyLabels: {
    explicit_label: number;
    proxy_signals: ProxySignal[];
    proxy_scores: {
      safety_score: number;
      vulnerability_score: number;
      complexity_score: number;
      security_score: number;
    };
    soft_labels: {
      vulnerable: number;
      safe: number;
      complex: number;
    };
    augmented_labels: {
      vulnerability_types: string[];
      severity_levels: string[];
      risk_factors: string[];
      security_patterns: string[];
    };
    confidence: number;
  };
  onToggleView?: (view: 'signals' | 'scores' | 'patterns') => void;
}

const ProxyLabelingInsights: React.FC<ProxyLabelingInsightsProps> = ({
  proxyLabels,
  onToggleView
}) => {
  const [activeView, setActiveView] = useState<'signals' | 'scores' | 'patterns'>('signals');
  const [showDetails, setShowDetails] = useState(false);
  const [filteredSignals, setFilteredSignals] = useState<ProxySignal[]>(proxyLabels.proxy_signals);

  // Filter signals by type
  const filterSignals = (type: string) => {
    if (type === 'all') {
      setFilteredSignals(proxyLabels.proxy_signals);
    } else {
      setFilteredSignals(proxyLabels.proxy_signals.filter(signal => signal.signal_type === type));
    }
  };

  // Get signal type color
  const getSignalTypeColor = (type: string) => {
    switch (type) {
      case 'security_pattern': return '#10b981';
      case 'safety_indicator': return '#3b82f6';
      case 'vulnerability_indicator': return '#ef4444';
      default: return '#6b7280';
    }
  };

  // Get signal type icon
  const getSignalTypeIcon = (type: string) => {
    switch (type) {
      case 'security_pattern': return Shield;
      case 'safety_indicator': return CheckCircle;
      case 'vulnerability_indicator': return AlertTriangle;
      default: return Info;
    }
  };

  // Prepare data for charts
  const signalTypeData = [
    { name: 'Security Patterns', value: proxyLabels.proxy_signals.filter(s => s.signal_type === 'security_pattern').length, color: '#10b981' },
    { name: 'Safety Indicators', value: proxyLabels.proxy_signals.filter(s => s.signal_type === 'safety_indicator').length, color: '#3b82f6' },
    { name: 'Vulnerability Indicators', value: proxyLabels.proxy_signals.filter(s => s.signal_type === 'vulnerability_indicator').length, color: '#ef4444' }
  ];

  const proxyScoresData = [
    { name: 'Safety', value: proxyLabels.proxy_scores.safety_score, color: '#10b981' },
    { name: 'Security', value: proxyLabels.proxy_scores.security_score, color: '#3b82f6' },
    { name: 'Complexity', value: proxyLabels.proxy_scores.complexity_score, color: '#8b5cf6' },
    { name: 'Vulnerability', value: proxyLabels.proxy_scores.vulnerability_score, color: '#ef4444' }
  ];

  const softLabelsData = [
    { name: 'Vulnerable', value: proxyLabels.soft_labels.vulnerable, color: '#ef4444' },
    { name: 'Safe', value: proxyLabels.soft_labels.safe, color: '#10b981' },
    { name: 'Complex', value: proxyLabels.soft_labels.complex, color: '#8b5cf6' }
  ];

  return (
    <div className="w-full bg-white/5 backdrop-blur-sm rounded-2xl border border-white/20 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-white/20">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-gradient-to-r from-green-500 to-blue-600 rounded-2xl">
            <Brain className="h-6 w-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Proxy Labeling Insights</h3>
            <p className="text-sm text-white/70">Advanced data labeling using security best practices</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300"
          >
            {showDetails ? <EyeOff className="h-5 w-5 text-white" /> : <Eye className="h-5 w-5 text-white" />}
          </button>
          <button className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300">
            <Info className="h-5 w-5 text-white" />
          </button>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="flex items-center space-x-1 p-4 border-b border-white/20">
        {[
          { id: 'signals', label: 'Proxy Signals', icon: Zap },
          { id: 'scores', label: 'Quality Scores', icon: BarChart3 },
          { id: 'patterns', label: 'Security Patterns', icon: Shield }
        ].map((tab) => (
          <button
            key={tab.id}
            onClick={() => {
              setActiveView(tab.id as any);
              onToggleView?.(tab.id as any);
            }}
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
      <div className="p-6">
        <AnimatePresence mode="wait">
          {activeView === 'signals' && (
            <motion.div
              key="signals"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Signal Type Distribution */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Signal Type Distribution</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={signalTypeData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {signalTypeData.map((entry, index) => (
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

              {/* Signal Filters */}
              <div className="flex items-center space-x-2">
                <Filter className="h-4 w-4 text-white/70" />
                <span className="text-sm text-white/70">Filter by type:</span>
                {['all', 'security_pattern', 'safety_indicator', 'vulnerability_indicator'].map((type) => (
                  <button
                    key={type}
                    onClick={() => filterSignals(type)}
                    className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-300 ${
                      filteredSignals === proxyLabels.proxy_signals.filter(s => type === 'all' || s.signal_type === type)
                        ? 'bg-white/20 text-white'
                        : 'bg-white/10 text-white/70 hover:text-white'
                    }`}
                  >
                    {type.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </button>
                ))}
              </div>

              {/* Signal List */}
              <div className="space-y-3">
                {filteredSignals.map((signal, index) => {
                  const IconComponent = getSignalTypeIcon(signal.signal_type);
                  return (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="bg-white/5 rounded-xl p-4 border border-white/10 hover:bg-white/10 transition-all duration-300"
                    >
                      <div className="flex items-start justify-between">
                        <div className="flex items-start space-x-3">
                          <div 
                            className="p-2 rounded-lg"
                            style={{ backgroundColor: getSignalTypeColor(signal.signal_type) + '20' }}
                          >
                            <IconComponent 
                              className="h-4 w-4" 
                              style={{ color: getSignalTypeColor(signal.signal_type) }}
                            />
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center space-x-2 mb-2">
                              <h5 className="font-semibold text-white">{signal.source_pattern}</h5>
                              <span 
                                className="px-2 py-1 rounded-full text-xs font-medium"
                                style={{ 
                                  backgroundColor: getSignalTypeColor(signal.signal_type) + '20',
                                  color: getSignalTypeColor(signal.signal_type)
                                }}
                              >
                                {signal.signal_type.replace('_', ' ')}
                              </span>
                            </div>
                            <p className="text-sm text-white/70 mb-2">{signal.context.description}</p>
                            <div className="flex items-center space-x-4">
                              <div className="flex items-center space-x-2">
                                <span className="text-xs text-white/60">Confidence:</span>
                                <div className="flex items-center space-x-2">
                                  <div className="w-16 bg-white/20 rounded-full h-2">
                                    <div 
                                      className="h-2 rounded-full"
                                      style={{ 
                                        width: `${signal.confidence * 100}%`,
                                        backgroundColor: getSignalTypeColor(signal.signal_type)
                                      }}
                                    />
                                  </div>
                                  <span className="text-xs text-white font-medium">
                                    {Math.round(signal.confidence * 100)}%
                                  </span>
                                </div>
                              </div>
                              <div className="flex items-center space-x-2">
                                <span className="text-xs text-white/60">Line:</span>
                                <span className="text-xs text-white font-medium">{signal.line_number}</span>
                              </div>
                              <div className="flex items-center space-x-2">
                                <span className="text-xs text-white/60">Weight:</span>
                                <span className="text-xs text-white font-medium">
                                  {Math.round(signal.importance_weight * 100)}%
                                </span>
                              </div>
                            </div>
                            {signal.context.vulnerability_types.length > 0 && (
                              <div className="mt-2">
                                <span className="text-xs text-white/60">Vulnerability Types:</span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {signal.context.vulnerability_types.map((type, idx) => (
                                    <span key={idx} className="px-2 py-1 bg-red-500/20 text-red-300 text-xs rounded-full">
                                      {type}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            </motion.div>
          )}

          {activeView === 'scores' && (
            <motion.div
              key="scores"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Proxy Scores Chart */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Proxy Quality Scores</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={proxyScoresData}>
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
                        {proxyScoresData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Soft Labels */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Soft Label Distribution</h4>
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={softLabelsData}
                        cx="50%"
                        cy="50%"
                        innerRadius={40}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {softLabelsData.map((entry, index) => (
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

              {/* Overall Confidence */}
              <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-xl p-4 border border-green-400/30">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-semibold text-white">Overall Proxy Confidence</h4>
                    <p className="text-sm text-white/70">Quality of proxy labeling signals</p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-white">
                      {Math.round(proxyLabels.confidence * 100)}%
                    </div>
                    <div className="text-sm text-white/70">High Quality</div>
                  </div>
                </div>
                <div className="mt-4">
                  <div className="w-full bg-white/20 rounded-full h-3">
                    <div 
                      className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full transition-all duration-1000"
                      style={{ width: `${proxyLabels.confidence * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeView === 'patterns' && (
            <motion.div
              key="patterns"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Security Patterns */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Detected Security Patterns</h4>
                <div className="grid grid-cols-2 gap-3">
                  {proxyLabels.augmented_labels.security_patterns.map((pattern, index) => (
                    <div key={index} className="bg-green-500/20 border border-green-400/30 rounded-lg p-3">
                      <div className="flex items-center space-x-2 mb-2">
                        <Shield className="h-4 w-4 text-green-300" />
                        <span className="text-sm font-medium text-green-300">{pattern}</span>
                      </div>
                      <p className="text-xs text-green-200">
                        Security best practice detected
                      </p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Vulnerability Types */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Vulnerability Types</h4>
                <div className="flex flex-wrap gap-2">
                  {proxyLabels.augmented_labels.vulnerability_types.map((type, index) => (
                    <span key={index} className="px-3 py-1 bg-red-500/20 text-red-300 text-sm rounded-full border border-red-400/30">
                      {type}
                    </span>
                  ))}
                </div>
              </div>

              {/* Risk Factors */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Risk Factors</h4>
                <div className="space-y-2">
                  {proxyLabels.augmented_labels.risk_factors.map((factor, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <AlertTriangle className="h-4 w-4 text-orange-400" />
                      <span className="text-sm text-white">{factor}</span>
                    </div>
                  ))}
                </div>
              </div>

              {/* Severity Levels */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Severity Assessment</h4>
                <div className="flex flex-wrap gap-2">
                  {proxyLabels.augmented_labels.severity_levels.map((level, index) => (
                    <span 
                      key={index} 
                      className={`px-3 py-1 text-sm rounded-full border ${
                        level === 'critical' 
                          ? 'bg-red-500/20 text-red-300 border-red-400/30'
                          : level === 'high'
                          ? 'bg-orange-500/20 text-orange-300 border-orange-400/30'
                          : level === 'medium'
                          ? 'bg-yellow-500/20 text-yellow-300 border-yellow-400/30'
                          : 'bg-green-500/20 text-green-300 border-green-400/30'
                      }`}
                    >
                      {level.toUpperCase()}
                    </span>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default ProxyLabelingInsights;
