"use client";

interface AnalyzeButtonProps {
  onClick: () => void;
  disabled?: boolean;
}

export default function AnalyzeButton({ onClick, disabled }: AnalyzeButtonProps) {
  return (
    <div className="flex flex-col items-center gap-3">
      <button
        onClick={onClick}
        disabled={disabled}
        className={`
          relative group px-10 py-3.5 rounded-full text-sm font-medium tracking-wide
          transition-all duration-200 cursor-pointer
          ${
            disabled
              ? "bg-white/5 text-white/20 border border-white/8 cursor-not-allowed"
              : "bg-amber-500 hover:bg-amber-400 active:bg-amber-600 text-black shadow-lg shadow-amber-500/20 hover:shadow-amber-400/30 hover:scale-[1.02] active:scale-[0.98]"
          }
        `}
      >
        {/* Shimmer effect when enabled */}
        {!disabled && (
          <span className="absolute inset-0 rounded-full overflow-hidden">
            <span className="absolute inset-0 -translate-x-full group-hover:translate-x-full transition-transform duration-700 bg-linear-to-r from-transparent via-white/20 to-transparent" />
          </span>
        )}
        <span className="relative z-10 flex items-center gap-2">
          Analyze Review
          {!disabled && (
            <span className="text-black/60 group-hover:translate-x-0.5 transition-transform duration-150">
              →
            </span>
          )}
        </span>
      </button>

      {disabled && (
        <p className="text-[11px] text-white/20 font-mono">
          Enter a review to continue
        </p>
      )}
    </div>
  );
}
