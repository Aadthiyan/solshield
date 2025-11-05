'use client';

import { useEffect } from 'react';
import { useAppStore } from '@/utils/store';

const SoundManager: React.FC = () => {
  const { isAnalyzing } = useAppStore();

  useEffect(() => {
    if (isAnalyzing) {
      // Play subtle analysis sound (optional)
      // You can add actual audio files here
      // const audio = new Audio('/sounds/analysis.mp3');
      // audio.volume = 0.1;
      // audio.play();
    }
  }, [isAnalyzing]);

  return null;
};

// Sound helper functions
export const playHoverSound = () => {
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 800;
    oscillator.type = 'sine';
    gainNode.gain.setValueAtTime(0.02, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.1);
  } catch (error) {
    // Ignore errors
  }
};

export const playClickSound = () => {
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.setValueAtTime(800, audioContext.currentTime);
    oscillator.frequency.exponentialRampToValueAtTime(600, audioContext.currentTime + 0.05);
    oscillator.type = 'sine';
    gainNode.gain.setValueAtTime(0.03, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.1);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.15);
  } catch (error) {
    // Ignore errors
  }
};

export const playSuccessSound = () => {
  try {
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const oscillators = [];
    
    for (let i = 0; i < 3; i++) {
      const oscillator = audioContext.createOscillator();
      const gainNode = audioContext.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioContext.destination);

      oscillator.frequency.value = 800 + (i * 200);
      oscillator.type = i === 0 ? 'sine' : 'triangle';
      gainNode.gain.setValueAtTime(0.02, audioContext.currentTime + i * 0.05);
      gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3 + i * 0.05);

      oscillator.start(audioContext.currentTime + i * 0.05);
      oscillator.stop(audioContext.currentTime + 0.3 + i * 0.05);
      oscillators.push(oscillator);
    }
  } catch (error) {
    // Ignore errors
  }
};

export default SoundManager;

