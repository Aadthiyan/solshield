'use client';

import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Terminal, Play } from 'lucide-react';

const TerminalWindow: React.FC = () => {
  const [commandHistory, setCommandHistory] = useState<string[]>([]);
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    const commands = [
      '$ npm install @smart-contract-analyzer',
      '$ scanalyzer --scan ./contract.sol',
      '$ Loading neural networks...',
      '$ ✓ Analysis complete',
      '> Ready to scan your contract'
    ];

    let index = 0;
    const interval = setInterval(() => {
      if (index < commands.length) {
        setCommandHistory([...commands.slice(0, index + 1)]);
        index++;
        if (index === commands.length) {
          setIsTyping(false);
          clearInterval(interval);
        }
      }
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div
      className="relative max-w-4xl mx-auto my-20"
      initial={{ opacity: 0, y: 50 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
    >
      {/* Glowing terminal window */}
      <div className="relative bg-black/80 backdrop-blur-sm border border-purple-500/40 rounded-2xl overflow-hidden shadow-2xl">
        {/* Terminal header */}
        <div className="flex items-center space-x-2 px-6 py-4 bg-purple-500/10 border-b border-purple-500/20">
          <div className="flex space-x-2">
            <div className="w-3 h-3 bg-red-500 rounded-full" />
            <div className="w-3 h-3 bg-yellow-500 rounded-full" />
            <div className="w-3 h-3 bg-green-500 rounded-full" />
          </div>
          <Terminal className="w-5 h-5 ml-4 text-purple-400" />
          <span className="text-sm font-mono text-purple-300 ml-2">terminal</span>
        </div>

        {/* Terminal body */}
        <div className="p-8 font-mono text-sm">
          <div className="mb-4">
            <span className="text-green-400">user@analyzer</span>
            <span className="text-purple-400">:</span>
            <span className="text-blue-400">~</span>
            <span className="text-purple-400">$</span>
          </div>

          <div className="space-y-2">
            {commandHistory.map((cmd, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="flex items-start"
              >
                {cmd.startsWith('$') || cmd.startsWith('>') ? (
                  <span className="text-cyan-400">{cmd}</span>
                ) : (
                  <span className="text-white">{cmd}</span>
                )}
              </motion.div>
            ))}

            {isTyping && (
              <motion.span
                className="text-purple-400"
                animate={{ opacity: [0, 1, 0] }}
                transition={{ duration: 1, repeat: Infinity }}
              >
                ▊
              </motion.span>
            )}
          </div>
        </div>

        {/* Launch button */}
        <div className="p-6 bg-black/40 border-t border-purple-500/20">
          <motion.button
            className="w-full flex items-center justify-center space-x-3 px-8 py-4 bg-gradient-to-r from-purple-500 to-blue-500 rounded-xl font-bold text-lg shadow-lg"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Play className="w-6 h-6" />
            <span>Launch Security Scan</span>
          </motion.button>
        </div>
      </div>

      {/* Starfield background */}
      <div className="absolute inset-0 overflow-hidden rounded-2xl pointer-events-none">
        {[...Array(100)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute w-0.5 h-0.5 bg-white rounded-full"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
            }}
            animate={{
              opacity: [0, 1, 0],
            }}
            transition={{
              duration: 2 + Math.random() * 2,
              repeat: Infinity,
              delay: Math.random() * 2,
            }}
          />
        ))}
      </div>
    </motion.div>
  );
};

export default TerminalWindow;

