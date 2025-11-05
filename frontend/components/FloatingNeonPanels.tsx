'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Shield, Brain, BarChart3 } from 'lucide-react';

const FloatingNeonPanels: React.FC = () => {
  const panels = [
    { icon: Shield, title: 'DETECTION', description: 'AI scans for vulnerabilities' },
    { icon: Brain, title: 'ANALYSIS', description: 'Deep learning analysis' },
    { icon: BarChart3, title: 'REPORTING', description: 'Detailed insights' },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8 my-20">
      {panels.map((panel, index) => {
        const Icon = panel.icon;
        return (
          <motion.div
            key={index}
            className="relative group"
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.2, duration: 0.8 }}
          >
            {/* Glowing border */}
            <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-purple-500/20 to-blue-500/20 blur-xl opacity-75 group-hover:opacity-100 transition-opacity" />
            
            {/* Main panel */}
            <div className="relative bg-black/50 backdrop-blur-sm border border-purple-500/30 rounded-2xl p-8 min-h-[250px] flex flex-col items-center justify-center group-hover:border-purple-500/60 transition-all">
              {/* Icon */}
              <motion.div
                className="mb-6"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 3, repeat: Infinity, delay: index * 0.5 }}
              >
                <div className="p-4 bg-gradient-to-r from-purple-500/20 to-blue-500/20 rounded-xl">
                  <Icon className="w-16 h-16 text-purple-400" />
                </div>
              </motion.div>

              {/* Title */}
              <h3 className="text-2xl font-bold mb-3 font-orbitron text-purple-300">
                {panel.title}
              </h3>

              {/* Description */}
              <p className="text-center text-sm opacity-80">
                {panel.description}
              </p>

              {/* Particle stream effect */}
              <div className="absolute inset-0 overflow-hidden rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity">
                {[...Array(20)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute w-1 h-1 bg-purple-400 rounded-full"
                    style={{
                      left: '50%',
                      top: `${Math.random() * 100}%`,
                    }}
                    animate={{
                      y: [0, -100],
                      opacity: [1, 0],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      delay: (i * 0.1),
                    }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};

export default FloatingNeonPanels;

