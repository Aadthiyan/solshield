'use client';

import React, { useEffect, useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Info, 
  AlertTriangle, 
  CheckCircle, 
  Code,
  GitBranch,
  Shield,
  Eye,
  EyeOff
} from 'lucide-react';

interface GraphNode {
  id: string;
  label: string;
  type: 'function' | 'variable' | 'statement' | 'expression' | 'modifier';
  syntax_subtree: any;
  semantic_features: {
    has_external_call: boolean;
    has_state_modification: boolean;
    has_loop: boolean;
    has_condition: boolean;
    security_patterns: string[];
    vulnerability_indicators: string[];
  };
  position: [number, number];
  importance_score: number;
  vulnerability_confidence: number;
  is_vulnerable: boolean;
}

interface GraphEdge {
  source: string;
  target: string;
  type: 'control_flow' | 'data_flow' | 'call_flow' | 'dependency';
  weight: number;
  semantic_context: any;
}

interface GraphVisualizationProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  contractName: string;
  onNodeClick?: (node: GraphNode) => void;
  onEdgeClick?: (edge: GraphEdge) => void;
  showVulnerabilities?: boolean;
  showSecurityPatterns?: boolean;
  showSemanticFlow?: boolean;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({
  nodes,
  edges,
  contractName,
  onNodeClick,
  onEdgeClick,
  showVulnerabilities = true,
  showSecurityPatterns = true,
  showSemanticFlow = true
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Color schemes for different node types
  const getNodeColor = (node: GraphNode) => {
    if (node.is_vulnerable) return '#ef4444'; // Red for vulnerable
    if (node.semantic_features.security_patterns.length > 0) return '#10b981'; // Green for secure
    if (node.type === 'function') return '#3b82f6'; // Blue for functions
    if (node.type === 'variable') return '#8b5cf6'; // Purple for variables
    return '#6b7280'; // Gray for others
  };

  // Edge colors based on type
  const getEdgeColor = (edge: GraphEdge) => {
    switch (edge.type) {
      case 'control_flow': return '#f59e0b';
      case 'data_flow': return '#06b6d4';
      case 'call_flow': return '#8b5cf6';
      case 'dependency': return '#6b7280';
      default: return '#6b7280';
    }
  };

  // Calculate node size based on importance
  const getNodeSize = (node: GraphNode) => {
    const baseSize = 20;
    const importanceMultiplier = node.importance_score * 2;
    return baseSize + importanceMultiplier;
  };

  // Handle zoom
  const handleZoom = (delta: number) => {
    setZoom(prev => Math.max(0.1, Math.min(5, prev + delta)));
  };

  // Handle pan
  const handlePan = (dx: number, dy: number) => {
    setPan(prev => ({
      x: prev.x + dx,
      y: prev.y + dy
    }));
  };

  // Handle mouse events
  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX, y: e.clientY });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      handlePan(dx, dy);
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Handle wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    handleZoom(delta);
  };

  // Reset view
  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Filter nodes and edges based on visibility settings
  const filteredNodes = nodes.filter(node => {
    if (showVulnerabilities && node.is_vulnerable) return true;
    if (showSecurityPatterns && node.semantic_features.security_patterns.length > 0) return true;
    if (!showVulnerabilities && !showSecurityPatterns) return true;
    return false;
  });

  const filteredEdges = edges.filter(edge => {
    if (!showSemanticFlow) return false;
    return filteredNodes.some(node => node.id === edge.source || node.id === edge.target);
  });

  return (
    <div className="w-full h-full bg-white/5 backdrop-blur-sm rounded-2xl border border-white/20 shadow-2xl">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/20">
        <div className="flex items-center space-x-3">
          <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-xl">
            <GitBranch className="h-5 w-5 text-white" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Joint Syntax-Semantic Graph</h3>
            <p className="text-sm text-white/70">{contractName}</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <button
            onClick={() => handleZoom(0.1)}
            className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300"
          >
            <ZoomIn className="h-4 w-4 text-white" />
          </button>
          <button
            onClick={() => handleZoom(-0.1)}
            className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300"
          >
            <ZoomOut className="h-4 w-4 text-white" />
          </button>
          <button
            onClick={resetView}
            className="p-2 bg-white/10 hover:bg-white/20 rounded-lg transition-all duration-300"
          >
            <RotateCcw className="h-4 w-4 text-white" />
          </button>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between p-4 border-b border-white/20">
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2 text-sm text-white">
            <input
              type="checkbox"
              checked={showVulnerabilities}
              onChange={(e) => {/* Handle toggle */}}
              className="w-4 h-4 text-blue-600 bg-white/10 border-white/30 rounded focus:ring-blue-500"
            />
            <span>Show Vulnerabilities</span>
          </label>
          <label className="flex items-center space-x-2 text-sm text-white">
            <input
              type="checkbox"
              checked={showSecurityPatterns}
              onChange={(e) => {/* Handle toggle */}}
              className="w-4 h-4 text-blue-600 bg-white/10 border-white/30 rounded focus:ring-blue-500"
            />
            <span>Security Patterns</span>
          </label>
          <label className="flex items-center space-x-2 text-sm text-white">
            <input
              type="checkbox"
              checked={showSemanticFlow}
              onChange={(e) => {/* Handle toggle */}}
              className="w-4 h-4 text-blue-600 bg-white/10 border-white/30 rounded focus:ring-blue-500"
            />
            <span>Semantic Flow</span>
          </label>
        </div>
        
        <div className="text-sm text-white/70">
          Nodes: {filteredNodes.length} | Edges: {filteredEdges.length}
        </div>
      </div>

      {/* Graph Visualization */}
      <div className="relative h-96 overflow-hidden">
        <svg
          ref={svgRef}
          className="w-full h-full cursor-grab active:cursor-grabbing"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
          style={{ transform: `scale(${zoom}) translate(${pan.x}px, ${pan.y}px)` }}
        >
          {/* Edges */}
          {filteredEdges.map((edge, index) => {
            const sourceNode = filteredNodes.find(n => n.id === edge.source);
            const targetNode = filteredNodes.find(n => n.id === edge.target);
            
            if (!sourceNode || !targetNode) return null;
            
            return (
              <line
                key={index}
                x1={sourceNode.position[0]}
                y1={sourceNode.position[1]}
                x2={targetNode.position[0]}
                y2={targetNode.position[1]}
                stroke={getEdgeColor(edge)}
                strokeWidth={edge.weight * 3}
                opacity={0.6}
                className="cursor-pointer hover:opacity-100 transition-opacity duration-300"
                onClick={() => {
                  setSelectedEdge(edge);
                  onEdgeClick?.(edge);
                }}
              />
            );
          })}
          
          {/* Nodes */}
          {filteredNodes.map((node, index) => (
            <g key={node.id}>
              <circle
                cx={node.position[0]}
                cy={node.position[1]}
                r={getNodeSize(node)}
                fill={getNodeColor(node)}
                stroke={selectedNode?.id === node.id ? '#ffffff' : 'transparent'}
                strokeWidth={selectedNode?.id === node.id ? 2 : 0}
                className="cursor-pointer hover:opacity-80 transition-all duration-300"
                onClick={() => {
                  setSelectedNode(node);
                  onNodeClick?.(node);
                }}
              />
              
              {/* Node label */}
              <text
                x={node.position[0]}
                y={node.position[1] + 5}
                textAnchor="middle"
                className="text-xs fill-white font-medium pointer-events-none"
              >
                {node.label}
              </text>
              
              {/* Vulnerability indicator */}
              {node.is_vulnerable && (
                <circle
                  cx={node.position[0] + getNodeSize(node) - 5}
                  cy={node.position[1] - getNodeSize(node) + 5}
                  r={4}
                  fill="#ef4444"
                  className="animate-pulse"
                />
              )}
              
              {/* Security pattern indicator */}
              {node.semantic_features.security_patterns.length > 0 && (
                <circle
                  cx={node.position[0] - getNodeSize(node) + 5}
                  cy={node.position[1] - getNodeSize(node) + 5}
                  r={4}
                  fill="#10b981"
                />
              )}
            </g>
          ))}
        </svg>
      </div>

      {/* Node Details Panel */}
      <AnimatePresence>
        {selectedNode && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute bottom-4 left-4 right-4 bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 shadow-2xl"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg">
                  <Code className="h-4 w-4 text-white" />
                </div>
                <div>
                  <h4 className="font-bold text-white">{selectedNode.label}</h4>
                  <p className="text-sm text-white/70 capitalize">{selectedNode.type}</p>
                </div>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="p-1 hover:bg-white/20 rounded-lg transition-colors duration-300"
              >
                <EyeOff className="h-4 w-4 text-white/70" />
              </button>
            </div>
            
            <div className="grid grid-cols-2 gap-4 mb-3">
              <div>
                <p className="text-xs text-white/60 mb-1">Importance Score</p>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-white/20 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                      style={{ width: `${selectedNode.importance_score * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-white font-medium">
                    {Math.round(selectedNode.importance_score * 100)}%
                  </span>
                </div>
              </div>
              
              <div>
                <p className="text-xs text-white/60 mb-1">Vulnerability Confidence</p>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-white/20 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${
                        selectedNode.is_vulnerable 
                          ? 'bg-gradient-to-r from-red-500 to-red-600' 
                          : 'bg-gradient-to-r from-green-500 to-green-600'
                      }`}
                      style={{ width: `${selectedNode.vulnerability_confidence * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-white font-medium">
                    {Math.round(selectedNode.vulnerability_confidence * 100)}%
                  </span>
                </div>
              </div>
            </div>
            
            <div className="space-y-2">
              <div className="flex items-center space-x-2">
                {selectedNode.semantic_features.has_external_call && (
                  <span className="px-2 py-1 bg-orange-500/20 text-orange-300 text-xs rounded-full">
                    External Call
                  </span>
                )}
                {selectedNode.semantic_features.has_state_modification && (
                  <span className="px-2 py-1 bg-blue-500/20 text-blue-300 text-xs rounded-full">
                    State Modification
                  </span>
                )}
                {selectedNode.semantic_features.has_loop && (
                  <span className="px-2 py-1 bg-purple-500/20 text-purple-300 text-xs rounded-full">
                    Loop
                  </span>
                )}
                {selectedNode.semantic_features.has_condition && (
                  <span className="px-2 py-1 bg-cyan-500/20 text-cyan-300 text-xs rounded-full">
                    Condition
                  </span>
                )}
              </div>
              
              {selectedNode.semantic_features.security_patterns.length > 0 && (
                <div>
                  <p className="text-xs text-white/60 mb-1">Security Patterns</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedNode.semantic_features.security_patterns.map((pattern, index) => (
                      <span key={index} className="px-2 py-1 bg-green-500/20 text-green-300 text-xs rounded-full">
                        {pattern}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {selectedNode.semantic_features.vulnerability_indicators.length > 0 && (
                <div>
                  <p className="text-xs text-white/60 mb-1">Vulnerability Indicators</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedNode.semantic_features.vulnerability_indicators.map((indicator, index) => (
                      <span key={index} className="px-2 py-1 bg-red-500/20 text-red-300 text-xs rounded-full">
                        {indicator}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Edge Details Panel */}
      <AnimatePresence>
        {selectedEdge && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="absolute top-4 right-4 bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20 shadow-2xl max-w-sm"
          >
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-lg">
                  <GitBranch className="h-4 w-4 text-white" />
                </div>
                <div>
                  <h4 className="font-bold text-white capitalize">{selectedEdge.type.replace('_', ' ')}</h4>
                  <p className="text-sm text-white/70">Edge Connection</p>
                </div>
              </div>
              <button
                onClick={() => setSelectedEdge(null)}
                className="p-1 hover:bg-white/20 rounded-lg transition-colors duration-300"
              >
                <EyeOff className="h-4 w-4 text-white/70" />
              </button>
            </div>
            
            <div className="space-y-2">
              <div>
                <p className="text-xs text-white/60 mb-1">Connection Weight</p>
                <div className="flex items-center space-x-2">
                  <div className="flex-1 bg-white/20 rounded-full h-2">
                    <div 
                      className="bg-gradient-to-r from-cyan-500 to-blue-600 h-2 rounded-full"
                      style={{ width: `${selectedEdge.weight * 100}%` }}
                    />
                  </div>
                  <span className="text-xs text-white font-medium">
                    {Math.round(selectedEdge.weight * 100)}%
                  </span>
                </div>
              </div>
              
              <div>
                <p className="text-xs text-white/60 mb-1">Source → Target</p>
                <p className="text-sm text-white">
                  {selectedEdge.source} → {selectedEdge.target}
                </p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default GraphVisualization;
