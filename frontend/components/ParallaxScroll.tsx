'use client';

import { useEffect, useRef } from 'react';
import Lenis from 'lenis';
import { motion, useScroll, useTransform } from 'framer-motion';

export const SmoothScrollProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  useEffect(() => {
    const lenis = new Lenis({
      duration: 1.2,
      easing: (t) => Math.min(1, 1.001 - Math.pow(2, -10 * t)),
      smooth: true,
      smoothTouch: false,
    });

    function raf(time: number) {
      lenis.raf(time);
      requestAnimationFrame(raf);
    }

    requestAnimationFrame(raf);

    return () => {
      lenis.destroy();
    };
  }, []);

  return <>{children}</>;
};

interface ParallaxSectionProps {
  children: React.ReactNode;
  speed?: number;
  className?: string;
}

export const ParallaxSection: React.FC<ParallaxSectionProps> = ({ 
  children, 
  speed = 0.5, 
  className = '' 
}) => {
  const { scrollYProgress } = useScroll();
  const y = useTransform(scrollYProgress, [0, 1], ['0%', `${speed * 100}%`]);

  return (
    <motion.div
      className={className}
      style={{ y }}
    >
      {children}
    </motion.div>
  );
};

interface TiltOnHoverProps {
  children: React.ReactNode;
  intensity?: number;
}

export const TiltOnHover: React.FC<TiltOnHoverProps> = ({ 
  children, 
  intensity = 5 
}) => {
  return (
    <motion.div
      whileHover={{ 
        rotateX: intensity,
        rotateY: intensity,
        scale: 1.02,
      }}
      transition={{ duration: 0.3 }}
      style={{ perspective: '1000px' }}
    >
      {children}
    </motion.div>
  );
};

