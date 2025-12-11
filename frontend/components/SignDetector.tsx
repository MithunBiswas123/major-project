'use client';

import React, { useRef, useState, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';

interface Prediction {
  sign: string;
  confidence: number;
}

interface DetectionResult {
  hand_detected: boolean;
  prediction: string | null;
  confidence: number;
  all_predictions: Prediction[];
  landmarks: Array<Array<{x: number; y: number; z: number}>>;
}

export default function SignDetector() {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  
  const [isConnected, setIsConnected] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [result, setResult] = useState<DetectionResult | null>(null);
  const [signs, setSigns] = useState<string[]>([]);
  const [history, setHistory] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [lastSign, setLastSign] = useState<string | null>(null);

  // Fetch available signs on mount
  useEffect(() => {
    fetch('http://localhost:8000/signs')
      .then(res => res.json())
      .then(data => setSigns(data.signs))
      .catch(err => {
        console.error('Failed to fetch signs:', err);
        setError('Cannot connect to backend. Start the backend server first.');
      });
  }, []);

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket('ws://localhost:8000/ws/detect');
      
      ws.onopen = () => {
        console.log('WebSocket connected');
        setIsConnected(true);
        setError(null);
      };

      ws.onmessage = (event) => {
        try {
          const data: DetectionResult = JSON.parse(event.data);
          setResult(data);
          
          // Add to history if confident prediction and different from last
          if (data.prediction && data.confidence > 0.75) {
            if (data.prediction !== lastSign) {
              setLastSign(data.prediction);
              setHistory(prev => {
                const newHistory = [...prev, data.prediction!];
                return newHistory.slice(-15); // Keep last 15
              });
            }
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      };

      ws.onerror = () => {
        console.error('WebSocket error');
        setError('Connection error. Make sure the backend is running on port 8000.');
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (e) {
      setError('Failed to connect to backend server.');
    }
  }, [lastSign]);

  // Disconnect WebSocket
  const disconnect = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsConnected(false);
    setIsDetecting(false);
    setResult(null);
  }, []);

  // Capture and send frames
  useEffect(() => {
    if (!isDetecting) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      return;
    }

    intervalRef.current = setInterval(() => {
      if (webcamRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          wsRef.current.send(imageSrc);
        }
      }
    }, 150); // ~7 FPS for stability

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isDetecting]);

  // Draw landmarks on canvas
  useEffect(() => {
    if (!result?.landmarks || !canvasRef.current || !webcamRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const video = webcamRef.current.video;
    if (!video) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw landmarks for each hand
    result.landmarks.forEach((hand) => {
      // Draw connections
      const connections = [
        [0,1],[1,2],[2,3],[3,4], // Thumb
        [0,5],[5,6],[6,7],[7,8], // Index
        [0,9],[9,10],[10,11],[11,12], // Middle
        [0,13],[13,14],[14,15],[15,16], // Ring
        [0,17],[17,18],[18,19],[19,20], // Pinky
        [5,9],[9,13],[13,17] // Palm
      ];

      ctx.strokeStyle = '#00ff00';
      ctx.lineWidth = 2;

      connections.forEach(([i, j]) => {
        ctx.beginPath();
        ctx.moveTo(hand[i].x * canvas.width, hand[i].y * canvas.height);
        ctx.lineTo(hand[j].x * canvas.width, hand[j].y * canvas.height);
        ctx.stroke();
      });

      // Draw points
      hand.forEach((point) => {
        ctx.beginPath();
        ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, 2 * Math.PI);
        ctx.fillStyle = '#00ff00';
        ctx.fill();
      });
    });
  }, [result]);

  const startDetection = () => {
    connect();
    // Wait for connection before starting detection
    setTimeout(() => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        setIsDetecting(true);
      } else {
        // Retry connection
        connect();
        setTimeout(() => setIsDetecting(true), 1000);
      }
    }, 500);
  };

  const stopDetection = () => {
    setIsDetecting(false);
    setTimeout(() => disconnect(), 100);
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence > 0.8) return 'text-green-400';
    if (confidence > 0.5) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getConfidenceBarColor = (confidence: number) => {
    if (confidence > 0.8) return 'bg-green-500';
    if (confidence > 0.5) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Main Video Section */}
      <div className="lg:col-span-2">
        <div className="bg-slate-800 rounded-xl p-4 shadow-xl">
          {/* Video Container */}
          <div className="relative aspect-video bg-black rounded-lg overflow-hidden">
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              className="w-full h-full object-cover"
              mirrored={true}
              videoConstraints={{
                width: 640,
                height: 480,
                facingMode: 'user'
              }}
            />
            <canvas
              ref={canvasRef}
              className="absolute top-0 left-0 w-full h-full pointer-events-none"
              style={{ transform: 'scaleX(-1)' }}
            />
            
            {/* Prediction Overlay - Always show when detecting */}
            {isDetecting && (
              <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 bg-black/90 px-8 py-4 rounded-2xl border-2 border-green-500">
                {result?.hand_detected && result?.prediction ? (
                  <>
                    <span className={`text-4xl font-bold ${getConfidenceColor(result.confidence)}`}>
                      {result.prediction.toUpperCase()}
                    </span>
                    <span className="text-gray-300 ml-3 text-xl">
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                  </>
                ) : (
                  <span className="text-gray-400 text-xl">Show your hand...</span>
                )}
              </div>
            )}

            {/* Status Indicator */}
            <div className="absolute top-4 left-4">
              <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${
                isDetecting ? 'bg-green-500/20 text-green-400' : 'bg-gray-500/20 text-gray-400'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  isDetecting ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
                }`} />
                {isDetecting ? 'Detecting' : 'Stopped'}
              </div>
            </div>

            {/* Hand Status */}
            <div className="absolute top-4 right-4">
              <div className={`px-3 py-1 rounded-full ${
                result?.hand_detected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
              }`}>
                {result?.hand_detected ? '✋ Hand Detected' : '👋 Show Hand'}
              </div>
            </div>
          </div>

          {/* Controls */}
          <div className="flex gap-4 mt-4">
            {!isDetecting ? (
              <button
                onClick={startDetection}
                className="flex-1 bg-green-600 hover:bg-green-700 text-white py-3 px-6 rounded-lg font-semibold transition-colors"
              >
                🎬 Start Detection
              </button>
            ) : (
              <button
                onClick={stopDetection}
                className="flex-1 bg-red-600 hover:bg-red-700 text-white py-3 px-6 rounded-lg font-semibold transition-colors"
              >
                ⏹️ Stop Detection
              </button>
            )}
            <button
              onClick={clearHistory}
              className="bg-slate-700 hover:bg-slate-600 text-white py-3 px-6 rounded-lg font-semibold transition-colors"
            >
              🗑️ Clear
            </button>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mt-4 bg-red-500/20 border border-red-500 text-red-400 px-4 py-3 rounded-lg">
              ⚠️ {error}
            </div>
          )}
        </div>

        {/* Sentence Builder */}
        <div className="mt-4 bg-slate-800 rounded-xl p-4 shadow-xl">
          <h3 className="text-lg font-semibold mb-2 text-gray-300">📝 Detected Signs</h3>
          <div className="min-h-[50px] bg-slate-900 rounded-lg p-3 flex flex-wrap gap-2">
            {history.length === 0 ? (
              <span className="text-gray-500 italic">Signs will appear here...</span>
            ) : (
              history.map((sign, idx) => (
                <span
                  key={idx}
                  className="bg-blue-600 text-white px-3 py-1 rounded-full text-sm"
                >
                  {sign}
                </span>
              ))
            )}
          </div>
        </div>
      </div>

      {/* Sidebar */}
      <div className="space-y-4">
        {/* Top Predictions */}
        <div className="bg-slate-800 rounded-xl p-4 shadow-xl">
          <h3 className="text-lg font-semibold mb-3 text-gray-300">📊 Top Predictions</h3>
          <div className="space-y-2">
            {result?.all_predictions?.map((pred, idx) => (
              <div key={idx} className="flex items-center gap-3">
                <span className="w-20 text-sm font-medium truncate">{pred.sign}</span>
                <div className="flex-1 h-4 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${getConfidenceBarColor(pred.confidence)} transition-all duration-200`}
                    style={{ width: `${pred.confidence * 100}%` }}
                  />
                </div>
                <span className="text-sm text-gray-400 w-12 text-right">
                  {(pred.confidence * 100).toFixed(0)}%
                </span>
              </div>
            )) || (
              <p className="text-gray-500 italic text-sm">Start detection to see predictions</p>
            )}
          </div>
        </div>

        {/* Available Signs */}
        <div className="bg-slate-800 rounded-xl p-4 shadow-xl">
          <h3 className="text-lg font-semibold mb-3 text-gray-300">
            🤟 Available Signs ({signs.length})
          </h3>
          <div className="max-h-[300px] overflow-y-auto">
            <div className="flex flex-wrap gap-2">
              {signs.map((sign, idx) => (
                <span
                  key={idx}
                  className="bg-slate-700 text-gray-300 px-2 py-1 rounded text-xs hover:bg-slate-600 transition-colors"
                >
                  {sign}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="bg-slate-800 rounded-xl p-4 shadow-xl">
          <h3 className="text-lg font-semibold mb-3 text-gray-300">📖 How to Use</h3>
          <ol className="text-sm text-gray-400 space-y-2">
            <li>1. Click <strong className="text-green-400">Start Detection</strong></li>
            <li>2. Show your hand to the camera</li>
            <li>3. Make a sign gesture</li>
            <li>4. Hold steady for best results</li>
            <li>5. See the prediction appear!</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
