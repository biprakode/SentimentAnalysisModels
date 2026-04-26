"use client";

import StarRating from "./StarRating";

export interface ModelResult {
  model: "custom" | "hf" | "qwen" | "groq";
  rating: number;
}

const MODEL_META: Record<
  ModelResult["model"],
  { name: string; description: string; tag: string }
> = {
  custom: {
    name: "Custom Encoder SLM",
    description: "Trained on movie review dataset",
    tag: "Fine-tuned",
  },
  hf: {
    name: "Existing Encoder SLM",
    description: "Pretrained baseline model",
    tag: "Baseline",
  },
  qwen: {
    name: "Optimized Small LLM",
    description: "Fine-tuned with prompt optimization",
    tag: "Optimized",
  },
  groq: {
    name: "Standard LLM Response",
    description: "Basic prompting technique",
    tag: "Standard",
  },
};

interface ModelCardProps {
  result: ModelResult;
  index: number;
}

export default function ModelCard({ result, index }: ModelCardProps) {
  const meta = MODEL_META[result.model];

  return (
    <div
      className="group bg-white/3 border border-white/10 rounded-xl p-5 flex flex-col gap-4
                 hover:border-amber-500/30 transition-all duration-300 hover:bg-white/4.5"
      style={{ animationDelay: `${index * 80}ms` }}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-mono tracking-widest uppercase text-amber-400/60 bg-amber-500/8 border border-amber-500/15 px-2 py-0.5 rounded-full">
              {meta.tag}
            </span>
          </div>
          <h3 className="text-sm font-medium text-white/90 mt-0.5">{meta.name}</h3>
          <p className="text-xs text-white/30 font-light">{meta.description}</p>
        </div>

        {/* Big rating number */}
        <div className="flex flex-col items-end shrink-0">
          <span className="text-3xl font-semibold text-white/90 leading-none tabular-nums">
            {result.rating}
          </span>
          <span className="text-[10px] text-white/25 font-mono mt-0.5">/ 5</span>
        </div>
      </div>

      {/* Divider */}
      <div className="h-px bg-white/6" />

      {/* Stars + bar */}
      <div className="flex flex-col gap-2">
        <StarRating value={result.rating} size="sm" />

        {/* Visual bar */}
        <div className="h-1 w-full bg-white/5 rounded-full overflow-hidden">
          <div
            className="h-full bg-linear-to-r from-amber-500/70 to-amber-400/90 rounded-full transition-all duration-700"
            style={{ width: `${(result.rating / 5) * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
