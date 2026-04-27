"use client";

import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import axios from "axios";

import LoadingScreen from "@/components/LoadingScreen";
import ModelCard, { ModelResult } from "@/components/ModelCard";
import StarRating from "@/components/StarRating";
import MetricsTable from "@/components/MetricsTable";

export default function OutputPage() {
  const router = useRouter();

  const [loading, setLoading] = useState(true);
  const [results, setResults] = useState<any>(null);

  useEffect(() => {
    const stored = localStorage.getItem("analyzeData");
    if (!stored) return;

    const { review, mode, examples } = JSON.parse(stored);

    async function fetchData() {
      try {
        const res = await axios.post("/api/analyze", {
          review,
          mode,
          examples,
        });

        setResults(res.data);
      } catch (err) {
        console.error("Error:", err);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  // UI States
  const [userRating, setUserRating] = useState(0);
  const [submitted, setSubmitted] = useState(false);

  const stored =
    typeof window !== "undefined" ? localStorage.getItem("analyzeData") : null;
  const inputReview = stored ? JSON.parse(stored).review : "";

  const handleNewAnalysis = () => {
    localStorage.removeItem("analyzeData");
    router.push("/");
  };

  if (loading) {
    return <LoadingScreen />;
  }

  const modelResults: ModelResult[] = results?.results ?? [];
  const latencySeconds = results?.meta?.latency
    ? (results.meta.latency / 1000).toFixed(2)
    : null;

  return (
    <main className="min-h-screen bg-[#0c0e12] text-white px-4 py-16">
      {/* Ambient glow */}
      <div className="pointer-events-none fixed top-0 left-1/2 -translate-x-1/2 w-150 h-75 bg-amber-500/5 blur-[120px] rounded-full z-0" />
      {/* Film grain */}
      <div className="pointer-events-none fixed inset-0 z-0 opacity-[0.03] bg-[url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIzMDAiIGhlaWdodD0iMzAwIj48ZmlsdGVyIGlkPSJub2lzZSI+PGZlVHVyYnVsZW5jZSB0eXBlPSJmcmFjdGFsTm9pc2UiIGJhc2VGcmVxdWVuY3k9IjAuNjUiIG51bU9jdGF2ZXM9IjMiIHN0aXRjaFRpbGVzPSJzdGl0Y2giLz48ZmVDb2xvck1hdHJpeCB0eXBlPSJzYXR1cmF0ZSIgdmFsdWVzPSIwIi8+PC9maWx0ZXI+PHJlY3Qgd2lkdGg9IjMwMCIgaGVpZ2h0PSIzMDAiIGZpbHRlcj0idXJsKCNub2lzZSkiIG9wYWNpdHk9IjEiLz48L3N2Zz4=')]" />

      <div className="relative z-10 w-full max-w-2xl mx-auto flex flex-col gap-10">

        {/* ── Back nav ── */}
        <button
          onClick={handleNewAnalysis}
          className="mt-6 px-10 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 text-black text-sm font-medium transition-all duration-200 cursor-pointer shadow-[0_0_0_1px_rgba(251,191,36,0.2)] hover:shadow-[0_0_0_1px_rgba(251,191,36,0.4)]"
        >
          New Analysis
        </button>

        {/* ── Input Review Card ── */}
        <section className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <p className="text-[11px] font-mono tracking-widest uppercase text-white/30">
              Input Review
            </p>
            {latencySeconds && (
              <p className="text-[11px] font-mono text-white/25">
                Total Latency:{" "}
                <span className="text-amber-400/50">{latencySeconds}s</span>
              </p>
            )}
          </div>

          <div className="bg-white/3 border border-white/10 rounded-xl p-4">
            <p className="text-white/70 text-sm font-light leading-relaxed">
              {inputReview || (
                <span className="text-white/20 italic">No review found.</span>
              )}
            </p>
          </div>
        </section>

        {/* ── Divider ── */}
        <div className="h-px w-full bg-linear-to-r from-transparent via-white/10 to-transparent" />

        {/* ── Results Header ── */}
        <header className="text-center flex flex-col items-center gap-2">
          <div className="flex items-center gap-2 text-amber-400/70 text-[10px] tracking-[0.3em] uppercase font-mono">
            <span className="w-6 h-px bg-amber-400/35" />
            Complete
            <span className="w-6 h-px bg-amber-400/35" />
          </div>
          <h1 className="text-2xl font-semibold text-white/95 tracking-tight">
            Analysis Results
          </h1>
          <p className="text-xs text-white/30 font-light tracking-wide">
            Comparison across multiple models
          </p>
        </header>

        {/* ── 2x2 Model Grid ── */}
        {modelResults.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {modelResults.map((result, i) => (
              <ModelCard key={result.model} result={result} index={i} />
            ))}
          </div>
        ) : (
          <div className="bg-white/2 border border-white/8 rounded-xl p-8 text-center">
            <p className="text-white/25 text-sm font-light">
              No model results available.
            </p>
          </div>
        )}

        {/* ── Divider ── */}
        <div className="h-px w-full bg-linear-to-r from-transparent via-white/10 to-transparent" />

        {/* ── User Rating Section ── */}
        {!submitted ? (
          <section className="flex flex-col items-center gap-5">
            <div className="text-center flex flex-col gap-1.5">
              <p className="text-[11px] font-mono tracking-widest uppercase text-white/30">
                Your Rating
              </p>
              <p className="text-xs text-white/20 font-light">
                How would you rate this movie?
              </p>
            </div>

            <StarRating
              value={userRating}
              interactive
              onChange={setUserRating}
              size="lg"
            />

            <button
              onClick={() => userRating > 0 && setSubmitted(true)}
              disabled={userRating === 0}
              className={`px-8 py-3 rounded-full text-sm font-medium tracking-wide transition-all duration-200 cursor-pointer ${
                userRating > 0
                  ? "bg-amber-500 hover:bg-amber-400 text-black shadow-lg shadow-amber-500/20 hover:scale-[1.02] active:scale-[0.98]"
                  : "bg-white/5 text-white/20 border border-white/8 cursor-not-allowed"
              }`}
            >
              Submit Rating
            </button>
          </section>
        ) : (
          /* ── Metrics Table (after submit) ── */
          <MetricsTable results={modelResults} userRating={userRating} />
        )}

        {/* ── Footer ── */}
        <p className="text-center text-[11px] text-white/15 font-mono">
          No data is stored · Prompt Lens
        </p>
      </div>
    </main>
  );
}
