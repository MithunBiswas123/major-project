'use client';

import React, { useEffect, useState } from 'react';

const Hero: React.FC = () => {
  const [isVisible, setIsVisible] = useState<boolean>(false);

  useEffect(() => {
    setIsVisible(true);
  }, []);

  return (
    <div className="relative w-full min-h-[500px] overflow-hidden bg-black">
      {/* Animated Background Light Rays */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[400px] bg-white/20 rounded-full blur-[100px] animate-pulse" />
        <div className="absolute top-20 left-1/4 w-[400px] h-[400px] bg-white/15 rounded-full blur-[80px] animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-10 right-1/4 w-[300px] h-[300px] bg-white/15 rounded-full blur-[60px] animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      {/* Content */}
      <div className="relative z-10 flex flex-col items-center justify-center min-h-[500px] px-4 text-center">
        {/* Animated Badge */}
        <div
          className={`mb-6 transition-all duration-1000 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'
          }`}
        >
          <span className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-cyan-400 bg-cyan-400/10 border border-cyan-400/20 rounded-full">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500"></span>
            </span>
            AI-Powered Detection
          </span>
        </div>

        {/* Main Title */}
        <h1
          className={`text-5xl md:text-7xl font-bold mb-6 transition-all duration-1000 delay-200 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'
          }`}
        >
          <span className="bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-500 bg-clip-text text-transparent">
            Sign Language
          </span>
          <br />
          <span className="text-white">Detection</span>
        </h1>

        {/* Subtitle */}
        <p
          className={`text-xl md:text-2xl text-gray-400 max-w-2xl mb-8 transition-all duration-1000 delay-300 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'
          }`}
        >
          Real-time hand gesture recognition using{' '}
          <span className="text-cyan-400">TensorFlow</span> &{' '}
          <span className="text-purple-400">MediaPipe</span>
        </p>

        {/* Hand Gesture Icons */}
        <div
          className={`flex gap-4 mb-8 text-5xl md:text-6xl transition-all duration-1000 delay-500 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'
          }`}
        >
          <span className="animate-bounce" style={{ animationDelay: '0ms' }}>🤟</span>
          <span className="animate-bounce" style={{ animationDelay: '100ms' }}>✌️</span>
          <span className="animate-bounce" style={{ animationDelay: '200ms' }}>👍</span>
          <span className="animate-bounce" style={{ animationDelay: '300ms' }}>👋</span>
          <span className="animate-bounce" style={{ animationDelay: '400ms' }}>🖐️</span>
        </div>

        {/* CTA Button */}
        <div
          className={`transition-all duration-1000 delay-700 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 -translate-y-4'
          }`}
        >
          <a
            href="#detector"
            className="group relative px-8 py-4 bg-gradient-to-r from-cyan-500 to-blue-600 rounded-xl font-semibold text-white shadow-lg shadow-cyan-500/25 hover:shadow-cyan-500/50 transition-all duration-300 hover:scale-105 inline-flex items-center gap-2"
          >
            🎬 Start Detection
            <svg className="w-5 h-5 group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </a>
        </div>

        {/* Stats */}
        <div
          className={`grid grid-cols-3 gap-8 mt-12 transition-all duration-1000 delay-1000 ${
            isVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'
          }`}
        >
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-cyan-400">26</div>
            <div className="text-sm text-gray-500">Signs</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-purple-400">98%</div>
            <div className="text-sm text-gray-500">Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-3xl md:text-4xl font-bold text-blue-400">Real-time</div>
            <div className="text-sm text-gray-500">Detection</div>
          </div>
        </div>
      </div>

      {/* Scroll Indicator */}
      <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
        <svg className="w-6 h-6 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </div>
    </div>
  );
};

export default Hero;