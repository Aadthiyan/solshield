'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Brain, Link2, Cpu } from 'lucide-react';

const RotatingDodecahedron: React.FC = () => {
  const technologies = [
    { icon: Brain, label: 'AI' },
    { icon: Link2, label: 'BLOCKCHAIN' },
    { icon: Cpu, label: 'ML' },
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-12 my-20">
      {technologies.map((tech, index) => {
        const Icon = tech.icon;
        return (
          <motion.div
            key={index}
            className="relative w-64 h-64 mx-auto group"
            initial={{ opacity: 0, scale: 0.5 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.3, duration: 0.8 }}
            whileHover={{ scale: 1.1 }}
          >
            {/* 3D Dodecahedron effect */}
            <div className="relative w-full h-full">
              {/* Outer ring */}
              <motion.div
                className="absolute inset-0 border-4 border-purple-500/40 rounded-2xl"
                animate={{
                  rotateX: [0, 360],
                  rotateY: [0, 360],
                }}
                transition={{
                  rotateX: { duration: 20, repeat: Infinity, ease: 'linear' },
                  rotateY: { duration: 15, repeat: Infinity, ease: 'linear' },
                }}
                style={{ transformStyle: 'preserve-3d' }}
              />

              {/* Middle ring */}
              <motion.div
                className="absolute inset-4 border-3 border-blue-500/40 rounded-xl"
                animate={{
                  rotateX: [0, -360],
                  rotateY: [0, 360],
                }}
                transition={{
                  rotateX: { duration: 15, repeat: Infinity, ease: 'linear' },
                  rotateY: { duration: 20, repeat: Infinity, ease: 'linear' },
                }}
                style={{ transformStyle: 'preserve-3d' }}
              />

              {/* Inner core */}
              <motion.div
                className="absolute inset-8 bg-gradient-to-br from-purple-500/30 to-blue-500/30 backdrop-blur-sm rounded-lg flex items-center justify-center border-2 border-purple-500/30"
                animate={{
                  rotateZ: 360,
                }}
                transition={{
                  duration: 10,
                  repeat: Infinity,
                  ease: 'linear',
                }}
              >
                <Icon className="w-16 h-16 text-purple-300" />
              </motion.div>

              {/* Halo glow on hover */}
              <motion.div
                className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity"
                style={{
                  background: 'radial-gradient(circle, rgba(139, 92, 246, 0.3), transparent 70%)',
                  filter: 'blur(30px)',
                }}
              />
            </div>

            {/* Label */}
            <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2">
              <span className="text-xl font-bold font-orbitron text-purple-300">
                {tech.label}
              </span>
            </div>
          </motion.div>
        );
      })}
    </div>
  );
};

export default RotatingDodecahedron;

