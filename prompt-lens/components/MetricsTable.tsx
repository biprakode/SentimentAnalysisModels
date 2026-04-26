"use client";

import { ModelResult } from "./ModelCard";

const MODEL_NAMES: Record<ModelResult["model"], string> = {
  custom: "Custom Encoder SLM",
  hf: "Existing Encoder SLM",
  qwen: "Optimized Small LLM",
  groq: "Standard LLM Response",
};

interface MetricsTableProps {
  results: ModelResult[];
  userRating: number;
}

export default function MetricsTable({ results, userRating }: MetricsTableProps) {
  const sorted = [...results].sort(
    (a, b) =>
      Math.abs(a.rating - userRating) - Math.abs(b.rating - userRating)
  );

  return (
    <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-2 duration-500">
      <div className="flex items-center gap-3">
        <p className="text-[11px] font-mono tracking-widest uppercase text-white/30">
          Evaluation Metrics
        </p>
        <div className="h-px flex-1 bg-white/[0.06]" />
        <span className="text-[11px] font-mono text-white/20">
          vs. your rating · {userRating}/5
        </span>
      </div>

      <div className="overflow-hidden rounded-xl border border-white/8 bg-white/[0.02]">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/8">
              <th className="text-left px-4 py-3 text-[11px] font-mono tracking-widest uppercase text-white/25 font-normal">
                Model
              </th>
              <th className="text-center px-4 py-3 text-[11px] font-mono tracking-widest uppercase text-white/25 font-normal">
                Predicted
              </th>
              <th className="text-center px-4 py-3 text-[11px] font-mono tracking-widest uppercase text-white/25 font-normal">
                Actual
              </th>
              <th className="text-center px-4 py-3 text-[11px] font-mono tracking-widest uppercase text-white/25 font-normal">
                Error
              </th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r, i) => {
              const error = Math.abs(r.rating - userRating);
              const isFirst = i === 0;
              return (
                <tr
                  key={r.model}
                  className={`border-b border-white/[0.05] last:border-0 transition-colors duration-150 ${
                    isFirst ? "bg-amber-500/[0.04]" : "hover:bg-white/[0.02]"
                  }`}
                >
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      {isFirst && (
                        <span className="text-[9px] font-mono uppercase tracking-widest text-amber-400/70 bg-amber-500/10 border border-amber-500/20 px-1.5 py-0.5 rounded-full">
                          Best
                        </span>
                      )}
                      <span className="text-white/70 font-light text-xs">
                        {MODEL_NAMES[r.model]}
                      </span>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className="text-white/80 font-mono tabular-nums text-xs">
                      {r.rating}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className="text-white/80 font-mono tabular-nums text-xs">
                      {userRating}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span
                      className={`font-mono tabular-nums text-xs px-2 py-0.5 rounded-full ${
                        error === 0
                          ? "text-emerald-400/80 bg-emerald-500/10"
                          : error <= 1
                          ? "text-amber-400/80 bg-amber-500/10"
                          : "text-red-400/70 bg-red-500/10"
                      }`}
                    >
                      {error.toFixed(1)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
