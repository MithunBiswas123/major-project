'use client';

import SignDetector from '@/components/SignDetector';
import Hero from '@/components/Hero';
import SplineScene from '@/components/SplineScene';

export default function Home() {
  return (
    <>
      <Hero />
      
      {/* 3D Spline Scene */}
      <section className="py-8 px-4 bg-black">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-white mb-2">Interactive 3D Hand Model</h2>
            <p className="text-gray-400">Explore hand gestures in 3D</p>
          </div>
          <SplineScene />
        </div>
      </section>

      <main className="min-h-screen p-8 bg-slate-900">
        <div className="max-w-6xl mx-auto">
          {/* Header */}
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent">
              🤟 Sign Language Detection
            </h1>
            <p className="text-gray-400 mt-2">
              Real-time AI-powered sign language recognition
            </p>
          </div>

          {/* Main Component */}
          <SignDetector />

          {/* Footer */}
          <div className="mt-8 text-center text-gray-500 text-sm">
            <p>Powered by TensorFlow, MediaPipe & Next.js</p>
          </div>
        </div>
      </main>
    </>
  );
}
