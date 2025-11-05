'use client';

import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface CodeScanningAnimationProps {
  isActive?: boolean;
  className?: string;
}

const CodeScanningAnimation: React.FC<CodeScanningAnimationProps> = ({ 
  isActive = false, 
  className = '' 
}) => {
  const codeLines = [
    'function transfer(address to, uint amount) public {',
    '    require(balances[msg.sender] >= amount);',
    '    balances[msg.sender] -= amount;',
    '    balances[to] += amount;',
    '    emit Transfer(msg.sender, to, amount);',
    '}',
  ];

  return (
    <div className={`relative ${className}`}>
      {/* Scanning line effect */}
      <motion.div
        animate={{
          y: isActive ? ['0%', '100%'] : '0%',
        }}
        transition={{
          duration: 2,
          repeat: isActive ? Infinity : 0,
          repeatDelay: 0.5,
          ease: 'linear',
        }}
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'linear-gradient(to bottom, rgba(0, 255, 255, 0.1), rgba(0, 255, 255, 0.3))',
          height: '2px',
        }}
      />

      {/* Code lines */}
      <div className="font-mono text-sm text-cyan-300/50 space-y-1">
        {codeLines.map((line, index) => (
          <motion.div
            key={index}
            initial={{ opacity: 0.3, x: -20 }}
            animate={isActive ? { 
              opacity: [0.3, 1, 0.3],
              x: 0,
            } : { opacity: 0.3 }}
            transition={{
              duration: 1.5,
              delay: index * 0.2,
              repeat: isActive ? Infinity : 0,
              repeatDelay: 0.5,
            }}
          >
            {line}
          </motion.div>
        ))}
      </div>

      {/* Cyberpunk overlay */}
      <div className="absolute inset-0 pointer-events-none opacity-20">
        <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-b from-transparent via-cyan-500/10 to-transparent" />
        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-transparent via-cyan-500 to-transparent animate-pulse" />
      </div>
    </div>
  );
};

export default CodeScanningAnimation;

