"use client";

interface StarRatingProps {
  value: number;
  max?: number;
  interactive?: boolean;
  onChange?: (rating: number) => void;
  size?: "sm" | "md" | "lg";
}

export default function StarRating({
  value,
  max = 5,
  interactive = false,
  onChange,
  size = "md",
}: StarRatingProps) {
  const sizeClass = {
    sm: "text-base",
    md: "text-xl",
    lg: "text-2xl",
  }[size];

  return (
    <div className="flex items-center gap-1">
      {Array.from({ length: max }, (_, i) => i + 1).map((star) => (
        <button
          key={star}
          disabled={!interactive}
          onClick={() => interactive && onChange?.(star)}
          className={`${sizeClass} transition-all duration-100 leading-none ${
            interactive
              ? "cursor-pointer hover:scale-110 active:scale-95"
              : "cursor-default"
          } ${
            star <= Math.round(value)
              ? "text-amber-400"
              : "text-white/12"
          }`}
          aria-label={interactive ? `Rate ${star} stars` : `${star} star`}
        >
          ★
        </button>
      ))}
      <span className="ml-1 text-xs text-white/30 font-mono tabular-nums">
        {Number.isInteger(value) ? value : value.toFixed(1)}/{max}
      </span>
    </div>
  );
}
