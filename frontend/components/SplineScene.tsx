'use client';

import React, { Suspense, useEffect, useRef, useState } from 'react';

interface SplineSceneProps {
  scene?: string;
  className?: string;
}

const SplineScene: React.FC<SplineSceneProps> = ({ 
  // Default to a free Spline 3D scene - you can replace with your own
  scene = "https://prod.spline.design/bQ7kO30OIodv07em/scene.splinecode",
  className = "" 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let app: any = null;

    const loadSpline = async () => {
      try {
        // Dynamically import the runtime
        const { Application } = await import('@splinetool/runtime');
        
        if (canvasRef.current) {
          app = new Application(canvasRef.current);
          await app.load(scene);
          setLoading(false);
        }
      } catch (err) {
        console.error('Failed to load Spline scene:', err);
        setError('Failed to load 3D scene');
        setLoading(false);
      }
    };

    loadSpline();

    return () => {
      if (app) {
        app.dispose();
      }
    };
  }, [scene]);

  return (
    <div className={`relative w-full h-[400px] bg-black rounded-xl overflow-hidden ${className}`}>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500 mx-auto mb-4"></div>
            <p className="text-gray-400">Loading 3D Scene...</p>
          </div>
        </div>
      )}
      
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
          <div className="text-center text-red-400">
            <p>{error}</p>
          </div>
        </div>
      )}
      
      <canvas ref={canvasRef} className="w-full h-full" />
      
      {/* Overlay gradient for better integration */}
      <div className="absolute bottom-0 left-0 right-0 h-20 bg-gradient-to-t from-black to-transparent pointer-events-none" />
    </div>
  );
};

export default SplineScene;
