"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import ToggleSwitch from "@/components/ToggleSwitch";
import ReviewInput from "@/components/ReviewInput";
import FewShotExamples from "@/components/FewShotExamples";
import AnalyzeButton from "@/components/AnalyzeButton";

export type Mode = "zero-shot" | "few-shot";

export interface Example {
  id: string;
  review: string;
  rating: number;
}

export default function Home() {
  const router = useRouter();
  const [mode, setMode] = useState<Mode>("zero-shot");
  const [review, setReview] = useState("");
  const [examples, setExamples] = useState<Example[]>([
    { id: "1", review: "", rating: 0 },
  ]);

  const handleAnalyze = () => {
    const payload = {
      review,
      mode,
      examples,
    };

    localStorage.setItem("analyzeData", JSON.stringify(payload));
    router.push("/output");
  };

  return (
    <main className="min-h-screen bg-[#0c0e12] text-white flex flex-col items-center px-4 py-16">
      {/* Film grain overlay */}
      <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.03] bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48ZmlsdGVyIGlkPSJub2lzZSI+PGZlVHVyYnVsZW5jZSB0eXBlPSJmcmFjdGFsTm9pc2UiIGJhc2VGcmVxdWVuY3k9IjAuNjUiIG51bU9jdGF2ZXM9IjMiIHN0aXRjaFRpbGVzPSJzdGl0Y2giLz48ZmVDb2xvck1hdHJpeCB0eXBlPSJzYXR1cmF0ZSIgdmFsdWVzPSIwIi8+PC9maWx0ZXI+PHJlY3Qgd2lkdGg9IjMwMCIgaGVpZ2h0PSIzMDAiIGZpbHRlcj0idXJsKCNub2lzZSkiIG9wYWNpdHk9IjEiLz48L3N2Zz4=')]" />

      {/* Ambient glow */}
      <div className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-150 h-75 bg-amber-500/5 blur-[120px] rounded-full z-0" />

      <div className="relative z-10 w-full max-w-2xl flex flex-col gap-10">
        {/* Header */}
        <header className="text-center flex flex-col items-center gap-3">
          <div className="flex items-center gap-2 text-amber-400/80 text-xs tracking-[0.3em] uppercase font-mono mb-1">
            <span className="w-8 h-px bg-amber-400/40" />
            Prompt Lens
            <span className="w-8 h-px bg-amber-400/40" />
          </div>
          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white/95 leading-tight">
            🎬 Movie Review Analyzer
          </h1>
          <p className="text-sm text-white/35 font-light tracking-wide max-w-sm">
            Analyze sentiment and tone from any movie review using zero-shot or few-shot prompting
          </p>
        </header>

        {/* Divider */}
        <div className="h-px w-full bg-linear-to-r from-transparent via-white/10 to-transparent" />

        {/* Toggle */}
        <section className="flex flex-col items-center gap-3">
          <p className="text-[11px] text-white/30 tracking-widest uppercase font-mono">
            Prompt Mode
          </p>
          <ToggleSwitch mode={mode} onChange={setMode} />
        </section>

        {/* Review Input */}
        <ReviewInput value={review} onChange={setReview} />

        {/* Few-shot Examples */}
        {mode === "few-shot" && (
          <FewShotExamples examples={examples} onChange={setExamples} />
        )}

        {/* Analyze Button */}
        <AnalyzeButton onClick={handleAnalyze} disabled={!review.trim()} />

        <p className="text-center text-[11px] text-white/20 font-mono">
          No data is stored · Analysis is simulated
        </p>
      </div>
    </main>
  );
}
