"use client";

import { useEffect, useState } from "react";

export default function LoadingScreen() {
  const [dots, setDots] = useState(".");

  useEffect(() => {
    const frames = [".", "..", "..."];
    let i = 0;
    const interval = setInterval(() => {
      i = (i + 1) % frames.length;
      setDots(frames[i]);
    }, 500);
    return () => clearInterval(interval);
  }, []);

  return (
    <main className="min-h-screen bg-[#0c0e12] flex flex-col items-center justify-center gap-5">
      {/* Ambient glow */}
      <div className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-[500px] h-[260px] bg-amber-500/5 blur-[110px] rounded-full" />

      <div className="relative flex flex-col items-center gap-4">
        {/* Pulsing film icon */}
        <div className="relative">
          <div className="absolute inset-0 rounded-full bg-amber-500/10 animate-ping" />
          <div className="relative text-4xl animate-pulse">🎬</div>
        </div>

        <div className="flex flex-col items-center gap-1">
          <p className="text-white/70 text-sm font-light tracking-wide">
            Analyzing with multiple models
            <span className="inline-block w-6 text-amber-400/70 font-mono">{dots}</span>
          </p>
          <p className="text-white/20 text-xs font-mono tracking-widest uppercase">
            Please wait
          </p>
        </div>

        {/* Progress bar */}
        <div className="w-48 h-px bg-white/5 rounded-full overflow-hidden mt-2">
          <div className="h-full bg-gradient-to-r from-amber-500/40 to-amber-400/80 animate-[loading-bar_1.8s_ease-in-out_infinite]" />
        </div>
      </div>

      <style jsx>{`
        @keyframes loading-bar {
          0% { width: 0%; margin-left: 0%; }
          50% { width: 70%; margin-left: 15%; }
          100% { width: 0%; margin-left: 100%; }
        }
      `}</style>
    </main>
  );
}
