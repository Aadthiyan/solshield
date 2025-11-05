'use client';

import React, { useState } from 'react';
import { 
  Info, 
  HelpCircle, 
  BookOpen, 
  Lightbulb, 
  Target, 
  Shield, 
  Brain,
  GitBranch,
  Zap,
  CheckCircle,
  AlertTriangle,
  X,
  ChevronRight,
  ChevronDown,
  ExternalLink
} from 'lucide-react';

interface UserGuidanceProps {
  feature: 'joint-graph' | 'proxy-labeling' | 'robustness' | 'enhanced-metrics' | 'general';
  onClose?: () => void;
}

const UserGuidance: React.FC<UserGuidanceProps> = ({ feature, onClose }) => {
  const [activeSection, setActiveSection] = useState<string>('overview');
  const [expandedSections, setExpandedSections] = useState<Set<string>>(new Set(['overview']));
  const [isVisible, setIsVisible] = useState(true);

  const toggleSection = (section: string) => {
    const newExpanded = new Set(expandedSections);
    if (newExpanded.has(section)) {
      newExpanded.delete(section);
    } else {
      newExpanded.add(section);
    }
    setExpandedSections(newExpanded);
  };

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      if (onClose) onClose();
    }, 300);
  };

  const getFeatureContent = () => {
    switch (feature) {
      case 'joint-graph':
        return {
          title: 'Joint Syntax-Semantic Graph Learning',
          icon: GitBranch,
          description: 'Advanced graph-based analysis combining syntax and semantic relationships',
          sections: [
            {
              id: 'overview',
              title: 'Overview',
              content: 'Joint syntax-semantic graph learning combines detailed syntax trees (AST subtrees) with semantic relationships to create rich, hybrid graphs for vulnerability detection.',
              features: [
                'Syntax-semantic graph construction',
                'Multi-head attention between syntax and semantic features',
                'Hierarchical processing for complex patterns',
                'Enhanced vulnerability detection capabilities'
              ]
            },
            {
              id: 'how-it-works',
              title: 'How It Works',
              content: 'The system builds hybrid graphs that include both detailed syntax trees and semantic relationships like control flow and data flow.',
              steps: [
                'Parse Solidity code into Abstract Syntax Tree (AST)',
                'Extract syntax subtrees for each code element',
                'Identify semantic relationships (control flow, data flow, dependencies)',
                'Construct hybrid graph with syntax and semantic nodes',
                'Apply hierarchical GNN with attention mechanisms',
                'Generate vulnerability predictions with confidence scores'
              ]
            },
            {
              id: 'benefits',
              title: 'Benefits',
              content: 'This approach provides superior vulnerability detection by understanding both code structure and semantic meaning.',
              benefits: [
                'Improved detection of complex vulnerabilities',
                'Better understanding of code context',
                'Enhanced accuracy through joint analysis',
                'Captures subtle interactions between syntax and semantics'
              ]
            },
            {
              id: 'interpretation',
              title: 'How to Interpret Results',
              content: 'Understanding the graph visualization and metrics helps you identify potential security issues.',
              tips: [
                'Red nodes indicate vulnerable code elements',
                'Green nodes show secure patterns',
                'Edge thickness represents relationship strength',
                'Node size indicates importance score',
                'Attention weights show model focus areas'
              ]
            }
          ]
        };

      case 'proxy-labeling':
        return {
          title: 'Proxy Labeling System',
          icon: Brain,
          description: 'Innovative data labeling using security best practices and proxy signals',
          sections: [
            {
              id: 'overview',
              title: 'Overview',
              content: 'Proxy labeling uses security best practices to infer safer code regions, enhancing the dataset beyond explicit vulnerability labels.',
              features: [
                'Security pattern detection',
                'Proxy signal generation',
                'Soft label creation',
                'Data augmentation with proxy signals'
              ]
            },
            {
              id: 'security-patterns',
              title: 'Security Patterns Detected',
              content: 'The system identifies various security best practices that indicate safer code.',
              patterns: [
                { name: 'Checks-Effects-Interactions', score: 0.9, description: 'Proper state management pattern' },
                { name: 'Access Control Modifier', score: 0.8, description: 'Authorization checks' },
                { name: 'SafeMath Operations', score: 0.85, description: 'Overflow protection' },
                { name: 'Event Emission', score: 0.7, description: 'Transparency mechanism' },
                { name: 'Time Lock Mechanism', score: 0.8, description: 'Delayed execution pattern' },
                { name: 'Multi-sig Validation', score: 0.9, description: 'Multiple signature verification' },
                { name: 'Circuit Breaker', score: 0.8, description: 'Emergency stop mechanism' },
                { name: 'Withdrawal Pattern', score: 0.75, description: 'Secure withdrawal implementation' }
              ]
            },
            {
              id: 'proxy-signals',
              title: 'Proxy Signal Types',
              content: 'Different types of proxy signals provide insights into code quality and security.',
              signals: [
                { type: 'Security Pattern', color: '#10b981', description: 'Detected security best practices' },
                { type: 'Safety Indicator', color: '#3b82f6', description: 'General safety measures' },
                { type: 'Vulnerability Indicator', color: '#ef4444', description: 'Potential security risks' }
              ]
            },
            {
              id: 'interpretation',
              title: 'How to Interpret Results',
              content: 'Understanding proxy labels helps you assess code quality and security posture.',
              tips: [
                'High proxy signal quality indicates better code security',
                'Multiple security patterns suggest robust implementation',
                'Proxy confidence scores show reliability of signals',
                'Soft labels provide probabilistic security assessment'
              ]
            }
          ]
        };

      case 'robustness':
        return {
          title: 'Robustness & Adversarial Defense',
          icon: Shield,
          description: 'Model security and resistance to adversarial attacks',
          sections: [
            {
              id: 'overview',
              title: 'Overview',
              content: 'The system includes comprehensive adversarial training and defense mechanisms to protect against malicious attacks.',
              features: [
                'Adversarial sample generation',
                'Defense mechanism implementation',
                'Robustness evaluation',
                'Real-time threat monitoring'
              ]
            },
            {
              id: 'attack-types',
              title: 'Attack Types Handled',
              content: 'The system defends against various types of adversarial attacks.',
              attacks: [
                { type: 'Code Obfuscation', description: 'Variable renaming, dead code insertion', detection: '85%' },
                { type: 'Semantic Perturbation', description: 'Redundant checks, condition modification', detection: '78%' },
                { type: 'Gradient-based', description: 'Gradient-based perturbations', detection: '92%' },
                { type: 'Ensemble Attacks', description: 'Combined attack strategies', detection: '88%' }
              ]
            },
            {
              id: 'defense-mechanisms',
              title: 'Defense Mechanisms',
              content: 'Multiple layers of defense protect the model from adversarial attacks.',
              mechanisms: [
                { name: 'Input Sanitization', status: 'Active', effectiveness: '95%' },
                { name: 'Adversarial Detection', status: 'Active', effectiveness: '87%' },
                { name: 'Robust Inference', status: 'Active', effectiveness: '92%' },
                { name: 'Model Validation', status: 'Active', effectiveness: '89%' },
                { name: 'Rate Limiting', status: 'Partial', effectiveness: '75%' },
                { name: 'Threat Monitoring', status: 'Active', effectiveness: '90%' }
              ]
            },
            {
              id: 'metrics',
              title: 'Robustness Metrics',
              content: 'Key metrics indicate the model\'s resistance to adversarial attacks.',
              metrics: [
                { name: 'Overall Robustness', description: 'General resistance to attacks' },
                { name: 'Adversarial Accuracy', description: 'Performance under attack conditions' },
                { name: 'Detection Rate', description: 'Ability to identify adversarial samples' },
                { name: 'Defense Effectiveness', description: 'Success of defense mechanisms' },
                { name: 'Model Stability', description: 'Consistency under adversarial conditions' }
              ]
            }
          ]
        };

      case 'enhanced-metrics':
        return {
          title: 'Enhanced Evaluation Metrics',
          icon: Target,
          description: 'Comprehensive metrics for advanced model evaluation',
          sections: [
            {
              id: 'overview',
              title: 'Overview',
              content: 'Enhanced metrics provide detailed insights into model performance, robustness, and interpretability.',
              categories: [
                'Standard classification metrics',
                'Robustness evaluation metrics',
                'Proxy label quality assessment',
                'Model contribution analysis',
                'Interpretability metrics'
              ]
            },
            {
              id: 'standard-metrics',
              title: 'Standard Metrics',
              content: 'Traditional classification metrics for model performance evaluation.',
              metrics: [
                { name: 'Accuracy', description: 'Overall correctness of predictions' },
                { name: 'Precision', description: 'True positive rate among positive predictions' },
                { name: 'Recall', description: 'True positive rate among actual positives' },
                { name: 'F1-Score', description: 'Harmonic mean of precision and recall' },
                { name: 'AUC-ROC', description: 'Area under the ROC curve' },
                { name: 'AUC-PR', description: 'Area under the Precision-Recall curve' }
              ]
            },
            {
              id: 'robustness-metrics',
              title: 'Robustness Metrics',
              content: 'Metrics that evaluate model performance under adversarial conditions.',
              metrics: [
                { name: 'Adversarial Accuracy', description: 'Performance on adversarial samples' },
                { name: 'Detection Rate', description: 'Ability to detect adversarial inputs' },
                { name: 'Robustness Score', description: 'Overall resistance to attacks' },
                { name: 'False Positive Rate', description: 'Rate of incorrect positive predictions' }
              ]
            },
            {
              id: 'model-contribution',
              title: 'Model Contribution Analysis',
              content: 'Understanding how different model components contribute to predictions.',
              components: [
                { name: 'Joint GNN', description: 'Syntax-semantic graph analysis contribution' },
                { name: 'CodeBERT', description: 'Semantic code understanding contribution' },
                { name: 'GNN', description: 'Traditional graph analysis contribution' },
                { name: 'Fusion', description: 'Effectiveness of model combination' }
              ]
            }
          ]
        };

      default:
        return {
          title: 'Smart Contract Vulnerability Detection',
          icon: Shield,
          description: 'Advanced AI-powered security analysis system',
          sections: [
            {
              id: 'overview',
              title: 'System Overview',
              content: 'This system uses state-of-the-art machine learning models to detect vulnerabilities in smart contracts.',
              features: [
                'Joint syntax-semantic graph learning',
                'Proxy labeling for enhanced data quality',
                'Adversarial training for robustness',
                'Comprehensive evaluation metrics'
              ]
            }
          ]
        };
    }
  };

  const content = getFeatureContent();
  const IconComponent = content.icon;

  if (!isVisible) return null;

  return (
    <div 
      className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4 animate-fadeIn"
      style={{
        animation: 'fadeIn 0.3s ease-in-out'
      }}
    >
      <div 
        className="w-full max-w-4xl max-h-[90vh] bg-white/10 backdrop-blur-lg rounded-2xl border border-white/20 shadow-2xl overflow-hidden animate-scaleIn"
        style={{
          animation: 'scaleIn 0.3s ease-in-out'
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-white/20">
          <div className="flex items-center space-x-3">
            <div className="p-3 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
              <IconComponent className="h-6 w-6 text-white" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">{content.title}</h2>
              <p className="text-sm text-white/70">{content.description}</p>
            </div>
          </div>
          {onClose && (
            <button
              onClick={handleClose}
              className="p-2 hover:bg-white/20 rounded-lg transition-colors duration-300"
            >
              <X className="h-5 w-5 text-white" />
            </button>
          )}
        </div>

        {/* Content */}
        <div className="flex h-[70vh]">
          {/* Sidebar Navigation */}
          <div className="w-64 bg-white/5 border-r border-white/20 p-4 overflow-y-auto">
            <nav className="space-y-2">
              {content.sections.map((section) => (
                <button
                  key={section.id}
                  onClick={() => setActiveSection(section.id)}
                  className={`w-full flex items-center justify-between p-3 rounded-lg text-left transition-all duration-300 ${
                    activeSection === section.id
                      ? 'bg-white/20 text-white'
                      : 'text-white/70 hover:text-white hover:bg-white/10'
                  }`}
                >
                  <span className="text-sm font-medium">{section.title}</span>
                  <ChevronRight className="h-4 w-4" />
                </button>
              ))}
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            {content.sections.map((section) => (
              activeSection === section.id && (
                <div 
                  key={section.id}
                  className="space-y-6 animate-slideIn"
                  style={{
                    animation: 'slideIn 0.3s ease-in-out'
                  }}
                >
                  <div>
                    <h3 className="text-lg font-semibold text-white mb-3">{section.title}</h3>
                    <p className="text-white/80 leading-relaxed">{section.content}</p>
                  </div>

                  {/* Features */}
                  {section.features && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Key Features</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {section.features.map((feature, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <CheckCircle className="h-4 w-4 text-green-400 flex-shrink-0" />
                            <span className="text-sm text-white/80">{feature}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Categories */}
                  {section.categories && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Categories</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {section.categories.map((category, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <CheckCircle className="h-4 w-4 text-blue-400 flex-shrink-0" />
                            <span className="text-sm text-white/80">{category}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Steps */}
                  {section.steps && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">How It Works</h4>
                      <div className="space-y-2">
                        {section.steps.map((step, index) => (
                          <div key={index} className="flex items-start space-x-3">
                            <div className="w-6 h-6 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white text-xs font-bold flex-shrink-0 mt-0.5">
                              {index + 1}
                            </div>
                            <span className="text-sm text-white/80">{step}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Benefits */}
                  {section.benefits && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Benefits</h4>
                      <div className="space-y-2">
                        {section.benefits.map((benefit, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <div className="w-2 h-2 bg-green-400 rounded-full flex-shrink-0 mt-2" />
                            <span className="text-sm text-white/80">{benefit}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Tips */}
                  {section.tips && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Interpretation Tips</h4>
                      <div className="space-y-2">
                        {section.tips.map((tip, index) => (
                          <div key={index} className="flex items-start space-x-2">
                            <Lightbulb className="h-4 w-4 text-yellow-400 flex-shrink-0 mt-0.5" />
                            <span className="text-sm text-white/80">{tip}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Patterns */}
                  {section.patterns && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Security Patterns</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {section.patterns.map((pattern, index) => (
                          <div key={index} className="bg-white/5 rounded-lg p-3 border border-white/10">
                            <div className="flex items-center justify-between mb-2">
                              <span className="font-medium text-white">{pattern.name}</span>
                              <span className="text-xs text-white/60">Score: {pattern.score}</span>
                            </div>
                            <p className="text-sm text-white/70">{pattern.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Signals */}
                  {section.signals && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Signal Types</h4>
                      <div className="space-y-2">
                        {section.signals.map((signal, index) => (
                          <div key={index} className="flex items-center space-x-3">
                            <div 
                              className="w-3 h-3 rounded-full"
                              style={{ backgroundColor: signal.color }}
                            />
                            <span className="text-sm text-white/80">{signal.type}</span>
                            <span className="text-xs text-white/60">{signal.description}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Attacks */}
                  {section.attacks && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Attack Types</h4>
                      <div className="space-y-2">
                        {section.attacks.map((attack, index) => (
                          <div key={index} className="bg-white/5 rounded-lg p-3 border border-white/10">
                            <div className="flex items-center justify-between mb-1">
                              <span className="font-medium text-white">{attack.type}</span>
                              <span className="text-xs text-green-300">Detection: {attack.detection}</span>
                            </div>
                            <p className="text-sm text-white/70">{attack.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Mechanisms */}
                  {section.mechanisms && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Defense Mechanisms</h4>
                      <div className="space-y-2">
                        {section.mechanisms.map((mechanism, index) => (
                          <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
                            <div>
                              <span className="font-medium text-white">{mechanism.name}</span>
                              <span className={`ml-2 px-2 py-1 rounded-full text-xs ${
                                mechanism.status === 'Active' 
                                  ? 'bg-green-500/20 text-green-300' 
                                  : 'bg-yellow-500/20 text-yellow-300'
                              }`}>
                                {mechanism.status}
                              </span>
                            </div>
                            <span className="text-sm text-white/70">{mechanism.effectiveness}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Metrics */}
                  {section.metrics && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Metrics</h4>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {section.metrics.map((metric, index) => (
                          <div key={index} className="bg-white/5 rounded-lg p-3 border border-white/10">
                            <span className="font-medium text-white">{metric.name}</span>
                            <p className="text-sm text-white/70 mt-1">{metric.description}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Components */}
                  {section.components && (
                    <div className="space-y-3">
                      <h4 className="text-md font-semibold text-white">Model Components</h4>
                      <div className="space-y-2">
                        {section.components.map((component, index) => (
                          <div key={index} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10">
                            <span className="font-medium text-white">{component.name}</span>
                            <span className="text-sm text-white/70">{component.description}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-white/20 bg-white/5">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Info className="h-4 w-4 text-white/60" />
              <span className="text-sm text-white/60">Need more help?</span>
            </div>
            <button className="text-sm text-blue-300 hover:text-blue-200 transition-colors duration-300">
              View Documentation <ExternalLink className="h-3 w-3 inline ml-1" />
            </button>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes fadeIn {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }
        
        @keyframes scaleIn {
          from {
            opacity: 0;
            transform: scale(0.9);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        
        @keyframes slideIn {
          from {
            opacity: 0;
            transform: translateX(20px);
          }
          to {
            opacity: 1;
            transform: translateX(0);
          }
        }
      `}</style>
    </div>
  );
};

export default UserGuidance;