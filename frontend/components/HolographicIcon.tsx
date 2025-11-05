'use client';

import React from 'react';
import { motion } from 'framer-motion';

const HolographicIcon: React.FC<{ visible?: boolean }> = ({ visible = true }) => {
  if (!visible) return null;

  return (
    <div className="w-full h-[400px] relative flex items-center justify-center">
      {/* Outer holographic ring */}
      <motion.div
        className="absolute w-64 h-64 border-4 border-cyan-500 rounded-full"
        style={{ boxShadow: '0 0 30px rgba(0, 255, 255, 0.5)' }}
        animate={{
          rotate: 360,
          scale: [1, 1.1, 1],
        }}
        transition={{
          rotate: { duration: 20, repeat: Infinity, ease: 'linear' },
          scale: { duration: 3, repeat: Infinity, ease: 'easeInOut' },
        }}
      />

      {/* Middle ring */}
      <motion.div
        className="absolute w-48 h-48 border-2 border-cyan-400 rounded-full"
        style={{ boxShadow: '0 0 20px rgba(0, 255, 255, 0.3)' }}
        animate={{
          rotate: -360,
          opacity: [0.5, 1, 0.5],
        }}
        transition={{
          rotate: { duration: 15, repeat: Infinity, ease: 'linear' },
          opacity: { duration: 2, repeat: Infinity, ease: 'easeInOut' },
        }}
      />

      {/* Inner contract icon */}
      <motion.div
        className="relative w-32 h-32 bg-gradient-to-br from-cyan-500/20 to-blue-500/20 backdrop-blur-sm rounded-2xl flex items-center justify-center border-2 border-cyan-500/30"
        animate={{
          y: [0, 10, 0],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
      >
        <svg className="w-20 h-20 text-cyan-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      </motion.div>

      {/* Animated particles */}
      {[...Array(8)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-2 h-2 bg-cyan-400 rounded-full"
          animate={{
            rotate: 360,
            x: Math.cos((i * Math.PI) / 4) * 100,
            y: Math.sin((i * Math.PI) / 4) * 100,
            opacity: [0.3, 1, 0.3],
          }}
          transition={{
            rotate: { duration: 10, repeat: Infinity, ease: 'linear' },
            x: { duration: 4, repeat: Infinity, ease: 'easeInOut', delay: i * 0.1 },
            y: { duration: 4, repeat: Infinity, ease: 'easeInOut', delay: i * 0.1 },
            opacity: { duration: 2, repeat: Infinity, ease: 'easeInOut', delay: i * 0.2 },
          }}
        />
      ))}
    </div>
  );
};

export default HolographicIcon;

