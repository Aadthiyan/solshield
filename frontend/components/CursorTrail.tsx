'use client';

import { useEffect } from 'react';

const CursorTrail: React.FC = () => {
  useEffect(() => {
    const trail: Array<{ x: number; y: number; timestamp: number }> = [];
    const maxTrailLength = 15;

    const handleMouseMove = (e: MouseEvent) => {
      trail.push({ x: e.clientX, y: e.clientY, timestamp: Date.now() });
      
      if (trail.length > maxTrailLength) {
        trail.shift();
      }

      updateTrail();
    };

    const updateTrail = () => {
      let existingTrail = document.getElementById('cursor-trail');
      if (!existingTrail) {
        existingTrail = document.createElement('div');
        existingTrail.id = 'cursor-trail';
        existingTrail.style.cssText = 'position: fixed; pointer-events: none; z-index: 9999;';
        document.body.appendChild(existingTrail);
      }

      // Clear and redraw trail
      existingTrail.innerHTML = '';

      trail.forEach((point, index) => {
        const age = Date.now() - point.timestamp;
        const opacity = Math.max(0, 1 - age / 500);
        const size = 10 - (age / 100);

        if (opacity > 0 && size > 0) {
          const dot = document.createElement('div');
          dot.style.cssText = `
            position: absolute;
            left: ${point.x}px;
            top: ${point.y}px;
            width: ${size}px;
            height: ${size}px;
            background: radial-gradient(circle, rgba(139, 92, 246, 0.8), rgba(79, 70, 229, 0.4));
            border-radius: 50%;
            opacity: ${opacity};
            transform: translate(-50%, -50%);
            pointer-events: none;
            box-shadow: 0 0 ${size * 3}px rgba(139, 92, 246, 0.6);
          `;
          existingTrail.appendChild(dot);
        }
      });

      // Remove old points
      while (trail.length > 0 && Date.now() - trail[0].timestamp > 500) {
        trail.shift();
      }
    };

    window.addEventListener('mousemove', handleMouseMove);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      const trailEl = document.getElementById('cursor-trail');
      if (trailEl) trailEl.remove();
    };
  }, []);

  return null;
};

export default CursorTrail;

