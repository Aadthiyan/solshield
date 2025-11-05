'use client';

import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Shield, 
  AlertTriangle, 
  CheckCircle, 
  Target, 
  Zap, 
  TrendingUp,
  TrendingDown,
  BarChart3,
  Info,
  Eye,
  EyeOff,
  RefreshCw,
  Lock,
  Unlock
} from 'lucide-react';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar
} from 'recharts';

interface AdversarialTest {
  attack_type: string;
  success_rate: number;
  detection_rate: number;
  robustness_score: number;
  samples_tested: number;
  confidence_drop: number;
}

interface RobustnessMetrics {
  overall_robustness: number;
  adversarial_accuracy: number;
  detection_rate: number;
  false_positive_rate: number;
  defense_effectiveness: number;
  model_stability: number;
}

interface RobustnessDashboardProps {
  robustnessMetrics: RobustnessMetrics;
  adversarialTests: AdversarialTest[];
  defenseStatus: 'active' | 'inactive' | 'partial';
  onToggleDefense?: (status: 'active' | 'inactive' | 'partial') => void;
  onRunTest?: (testType: string) => void;
}

const RobustnessDashboard: React.FC<RobustnessDashboardProps> = ({
  robustnessMetrics,
  adversarialTests,
  defenseStatus,
  onToggleDefense,
  onRunTest
}) => {
  const [activeTab, setActiveTab] = useState<'overview' | 'tests' | 'defense' | 'metrics'>('overview');
  const [showDetails, setShowDetails] = useState(false);
  const [isRunningTest, setIsRunningTest] = useState(false);

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Simulate real-time metric updates
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Get status color
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return '#10b981';
      case 'inactive': return '#ef4444';
      case 'partial': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  // Get attack type color
  const getAttackTypeColor = (type: string) => {
    switch (type) {
      case 'obfuscation': return '#8b5cf6';
      case 'semantic': return '#06b6d4';
      case 'gradient': return '#ef4444';
      case 'ensemble': return '#f59e0b';
      default: return '#6b7280';
    }
  };

  // Prepare data for charts
  const robustnessTrendData = [
    { time: '00:00', robustness: 85, attacks: 12 },
    { time: '01:00', robustness: 87, attacks: 8 },
    { time: '02:00', robustness: 89, attacks: 5 },
    { time: '03:00', robustness: 91, attacks: 3 },
    { time: '04:00', robustness: 88, attacks: 7 },
    { time: '05:00', robustness: 90, attacks: 4 },
    { time: '06:00', robustness: 92, attacks: 2 }
  ];

  const attackTypeData = adversarialTests.map(test => ({
    name: test.attack_type,
    success_rate: test.success_rate,
    detection_rate: test.detection_rate,
    robustness_score: test.robustness_score,
    color: getAttackTypeColor(test.attack_type)
  }));

  const radarData = [
    { metric: 'Robustness', value: robustnessMetrics.overall_robustness },
    { metric: 'Detection', value: robustnessMetrics.detection_rate },
    { metric: 'Accuracy', value: robustnessMetrics.adversarial_accuracy },
    { metric: 'Stability', value: robustnessMetrics.model_stability },
    { metric: 'Defense', value: robustnessMetrics.defense_effectiveness }
  ];

  const handleRunTest = async (testType: string) => {
    setIsRunningTest(true);
    onRunTest?.(testType);
    // Simulate test execution
    setTimeout(() => {
      setIsRunningTest(false);
    }, 3000);
  };

  return (
    <div className="w-full bg-white/5 backdrop-blur-sm rounded-2xl border border-white/20 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between p-6 border-b border-white/20">
        <div className="flex items-center space-x-3">
          <div className="p-3 bg-gradient-to-r from-red-500 to-orange-600 rounded-2xl">
            <Shield className="h-6 w-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white">Robustness & Adversarial Defense</h3>
            <p className="text-sm text-white/70">Model security and attack resistance analysis</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <div className="flex items-center space-x-2">
            <div 
              className={`w-3 h-3 rounded-full ${
                defenseStatus === 'active' ? 'bg-green-400' : 
                defenseStatus === 'partial' ? 'bg-yellow-400' : 'bg-red-400'
              }`}
            />
            <span className="text-sm text-white/70 capitalize">{defenseStatus} Defense</span>
          </div>
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
          { id: 'overview', label: 'Overview', icon: BarChart3 },
          { id: 'tests', label: 'Adversarial Tests', icon: Target },
          { id: 'defense', label: 'Defense Status', icon: Shield },
          { id: 'metrics', label: 'Detailed Metrics', icon: TrendingUp }
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
              {/* Key Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-xl p-4 border border-green-400/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white/70">Overall Robustness</p>
                      <p className="text-2xl font-bold text-white">
                        {Math.round(robustnessMetrics.overall_robustness)}%
                      </p>
                    </div>
                    <Shield className="h-8 w-8 text-green-300" />
                  </div>
                  <div className="mt-2">
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-green-500 to-blue-500 h-2 rounded-full transition-all duration-1000"
                        style={{ width: `${robustnessMetrics.overall_robustness}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-blue-500/20 to-purple-500/20 rounded-xl p-4 border border-blue-400/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white/70">Adversarial Accuracy</p>
                      <p className="text-2xl font-bold text-white">
                        {Math.round(robustnessMetrics.adversarial_accuracy)}%
                      </p>
                    </div>
                    <Target className="h-8 w-8 text-blue-300" />
                  </div>
                  <div className="mt-2">
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-1000"
                        style={{ width: `${robustnessMetrics.adversarial_accuracy}%` }}
                      />
                    </div>
                  </div>
                </div>

                <div className="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-4 border border-purple-400/30">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-white/70">Detection Rate</p>
                      <p className="text-2xl font-bold text-white">
                        {Math.round(robustnessMetrics.detection_rate)}%
                      </p>
                    </div>
                    <Zap className="h-8 w-8 text-purple-300" />
                  </div>
                  <div className="mt-2">
                    <div className="w-full bg-white/20 rounded-full h-2">
                      <div 
                        className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full transition-all duration-1000"
                        style={{ width: `${robustnessMetrics.detection_rate}%` }}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Robustness Trend */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Robustness Trend (24h)</h4>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={robustnessTrendData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                      <XAxis dataKey="time" stroke="rgba(255, 255, 255, 0.7)" />
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
                        dataKey="robustness" 
                        stroke="#10b981" 
                        strokeWidth={3}
                        dot={{ fill: '#10b981', strokeWidth: 2, r: 4 }}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {/* Defense Status */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-white">Defense Status</h4>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => onToggleDefense?.('active')}
                      className={`p-2 rounded-lg transition-all duration-300 ${
                        defenseStatus === 'active' 
                          ? 'bg-green-500/20 text-green-300 border border-green-400/30' 
                          : 'bg-white/10 text-white/70 hover:text-white'
                      }`}
                    >
                      <Lock className="h-4 w-4" />
                    </button>
                    <button
                      onClick={() => onToggleDefense?.('inactive')}
                      className={`p-2 rounded-lg transition-all duration-300 ${
                        defenseStatus === 'inactive' 
                          ? 'bg-red-500/20 text-red-300 border border-red-400/30' 
                          : 'bg-white/10 text-white/70 hover:text-white'
                      }`}
                    >
                      <Unlock className="h-4 w-4" />
                    </button>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white/70">Input Sanitization</span>
                      <span className="text-sm text-green-300">Active</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white/70">Adversarial Detection</span>
                      <span className="text-sm text-green-300">Active</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white/70">Robust Inference</span>
                      <span className="text-sm text-green-300">Active</span>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white/70">Rate Limiting</span>
                      <span className="text-sm text-yellow-300">Partial</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white/70">Model Validation</span>
                      <span className="text-sm text-green-300">Active</span>
                    </div>
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-white/70">Threat Monitoring</span>
                      <span className="text-sm text-green-300">Active</span>
                    </div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'tests' && (
            <motion.div
              key="tests"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Test Controls */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <div className="flex items-center justify-between mb-4">
                  <h4 className="text-lg font-semibold text-white">Adversarial Testing</h4>
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={() => handleRunTest('all')}
                      disabled={isRunningTest}
                      className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-300 disabled:opacity-50"
                    >
                      {isRunningTest ? (
                        <div className="flex items-center space-x-2">
                          <RefreshCw className="h-4 w-4 animate-spin" />
                          <span>Running...</span>
                        </div>
                      ) : (
                        'Run All Tests'
                      )}
                    </button>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {['obfuscation', 'semantic', 'gradient', 'ensemble'].map((testType) => (
                    <button
                      key={testType}
                      onClick={() => handleRunTest(testType)}
                      disabled={isRunningTest}
                      className="p-3 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300 disabled:opacity-50"
                    >
                      <div className="text-center">
                        <div 
                          className="w-8 h-8 mx-auto mb-2 rounded-lg flex items-center justify-center"
                          style={{ backgroundColor: getAttackTypeColor(testType) + '20' }}
                        >
                          <Target 
                            className="h-4 w-4" 
                            style={{ color: getAttackTypeColor(testType) }}
                          />
                        </div>
                        <span className="text-xs text-white capitalize">{testType}</span>
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Test Results */}
              <div className="space-y-4">
                {adversarialTests.map((test, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-white/5 rounded-xl p-4 border border-white/10"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center space-x-3">
                        <div 
                          className="p-2 rounded-lg"
                          style={{ backgroundColor: getAttackTypeColor(test.attack_type) + '20' }}
                        >
                          <Target 
                            className="h-5 w-5" 
                            style={{ color: getAttackTypeColor(test.attack_type) }}
                          />
                        </div>
                        <div>
                          <h5 className="font-semibold text-white capitalize">{test.attack_type} Attack</h5>
                          <p className="text-sm text-white/70">{test.samples_tested} samples tested</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-lg font-bold text-white">
                          {Math.round(test.robustness_score)}%
                        </div>
                        <div className="text-sm text-white/70">Robustness</div>
                      </div>
                    </div>
                    
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <p className="text-xs text-white/60 mb-1">Success Rate</p>
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-white/20 rounded-full h-2">
                            <div 
                              className="bg-red-500 h-2 rounded-full"
                              style={{ width: `${test.success_rate}%` }}
                            />
                          </div>
                          <span className="text-xs text-white font-medium">
                            {Math.round(test.success_rate)}%
                          </span>
                        </div>
                      </div>
                      
                      <div>
                        <p className="text-xs text-white/60 mb-1">Detection Rate</p>
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-white/20 rounded-full h-2">
                            <div 
                              className="bg-green-500 h-2 rounded-full"
                              style={{ width: `${test.detection_rate}%` }}
                            />
                          </div>
                          <span className="text-xs text-white font-medium">
                            {Math.round(test.detection_rate)}%
                          </span>
                        </div>
                      </div>
                      
                      <div>
                        <p className="text-xs text-white/60 mb-1">Confidence Drop</p>
                        <div className="flex items-center space-x-2">
                          <div className="flex-1 bg-white/20 rounded-full h-2">
                            <div 
                              className="bg-yellow-500 h-2 rounded-full"
                              style={{ width: `${test.confidence_drop}%` }}
                            />
                          </div>
                          <span className="text-xs text-white font-medium">
                            {Math.round(test.confidence_drop)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          )}

          {activeTab === 'defense' && (
            <motion.div
              key="defense"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Defense Mechanisms */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Active Defenses</h4>
                  <div className="space-y-3">
                    {[
                      { name: 'Input Sanitization', status: 'active', effectiveness: 95 },
                      { name: 'Adversarial Detection', status: 'active', effectiveness: 87 },
                      { name: 'Robust Inference', status: 'active', effectiveness: 92 },
                      { name: 'Model Validation', status: 'active', effectiveness: 89 }
                    ].map((defense, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <div className="w-3 h-3 rounded-full bg-green-400" />
                          <span className="text-sm text-white">{defense.name}</span>
                        </div>
                        <div className="text-sm text-green-300">{defense.effectiveness}%</div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Defense Effectiveness</h4>
                  <div className="h-48">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="rgba(255, 255, 255, 0.1)" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255, 255, 255, 0.7)', fontSize: 12 }} />
                        <PolarRadiusAxis tick={{ fill: 'rgba(255, 255, 255, 0.7)', fontSize: 10 }} />
                        <Radar
                          name="Defense Metrics"
                          dataKey="value"
                          stroke="#10b981"
                          fill="#10b981"
                          fillOpacity={0.3}
                          strokeWidth={2}
                        />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              {/* Threat Monitoring */}
              <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                <h4 className="text-lg font-semibold text-white mb-4">Threat Monitoring</h4>
                <div className="space-y-3">
                  {[
                    { threat: 'Code Obfuscation', level: 'medium', count: 12, blocked: 10 },
                    { threat: 'Semantic Perturbation', level: 'low', count: 8, blocked: 8 },
                    { threat: 'Gradient Attacks', level: 'high', count: 5, blocked: 4 },
                    { threat: 'Ensemble Attacks', level: 'medium', count: 7, blocked: 6 }
                  ].map((threat, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                      <div className="flex items-center space-x-3">
                        <div 
                          className={`w-3 h-3 rounded-full ${
                            threat.level === 'high' ? 'bg-red-400' :
                            threat.level === 'medium' ? 'bg-yellow-400' : 'bg-green-400'
                          }`}
                        />
                        <div>
                          <span className="text-sm text-white">{threat.threat}</span>
                          <span className="text-xs text-white/60 ml-2 capitalize">{threat.level}</span>
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <span className="text-sm text-white/70">{threat.count} attempts</span>
                        <span className="text-sm text-green-300">{threat.blocked} blocked</span>
                        <div className="w-16 bg-white/20 rounded-full h-2">
                          <div 
                            className="bg-green-500 h-2 rounded-full"
                            style={{ width: `${(threat.blocked / threat.count) * 100}%` }}
                          />
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          )}

          {activeTab === 'metrics' && (
            <motion.div
              key="metrics"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="space-y-6"
            >
              {/* Detailed Metrics */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Attack Performance</h4>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={attackTypeData}>
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
                        <Bar dataKey="success_rate" fill="#ef4444" name="Success Rate" />
                        <Bar dataKey="detection_rate" fill="#10b981" name="Detection Rate" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white/5 rounded-xl p-4 border border-white/10">
                  <h4 className="text-lg font-semibold text-white mb-4">Model Stability</h4>
                  <div className="space-y-4">
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-white/70">Overall Stability</span>
                        <span className="text-sm text-white font-medium">
                          {Math.round(robustnessMetrics.model_stability)}%
                        </span>
                      </div>
                      <div className="w-full bg-white/20 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-green-500 to-blue-500 h-3 rounded-full"
                          style={{ width: `${robustnessMetrics.model_stability}%` }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-white/70">False Positive Rate</span>
                        <span className="text-sm text-white font-medium">
                          {Math.round(robustnessMetrics.false_positive_rate)}%
                        </span>
                      </div>
                      <div className="w-full bg-white/20 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-red-500 to-orange-500 h-3 rounded-full"
                          style={{ width: `${robustnessMetrics.false_positive_rate}%` }}
                        />
                      </div>
                    </div>
                    
                    <div>
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm text-white/70">Defense Effectiveness</span>
                        <span className="text-sm text-white font-medium">
                          {Math.round(robustnessMetrics.defense_effectiveness)}%
                        </span>
                      </div>
                      <div className="w-full bg-white/20 rounded-full h-3">
                        <div 
                          className="bg-gradient-to-r from-purple-500 to-pink-500 h-3 rounded-full"
                          style={{ width: `${robustnessMetrics.defense_effectiveness}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Performance Summary */}
              <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 rounded-xl p-4 border border-green-400/30">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="text-lg font-semibold text-white">Security Assessment</h4>
                    <p className="text-sm text-white/70">Model demonstrates strong resistance to adversarial attacks</p>
                  </div>
                  <div className="text-right">
                    <div className="text-3xl font-bold text-white">
                      {robustnessMetrics.overall_robustness > 80 ? 'A+' : 
                       robustnessMetrics.overall_robustness > 70 ? 'A' : 
                       robustnessMetrics.overall_robustness > 60 ? 'B' : 'C'}
                    </div>
                    <div className="text-sm text-white/70">Security Grade</div>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default RobustnessDashboard;
