"use client";

import { Example } from "@/app/page";

interface ExampleCardProps {
  example: Example;
  index: number;
  onChange: (updated: Example) => void;
  onRemove: () => void;
  canRemove: boolean;
}

export default function ExampleCard({
  example,
  index,
  onChange,
  onRemove,
  canRemove,
}: ExampleCardProps) {
  const handleReviewChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange({ ...example, review: e.target.value });
  };

  const handleRatingChange = (rating: number) => {
    onChange({ ...example, rating });
  };

  return (
    <div className="relative bg-white/3 border border-white/8 rounded-xl p-5 flex flex-col gap-4 group/card hover:border-white/15 transition-colors duration-200">
      {/* Card header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="w-5 h-5 rounded-full bg-amber-500/15 border border-amber-500/30 text-amber-400 text-[10px] font-mono flex items-center justify-center">
            {index + 1}
          </span>
          <span className="text-[11px] text-white/40 tracking-widest uppercase font-mono">
            Example {index + 1}
          </span>
        </div>

        {canRemove && (
          <button
            onClick={onRemove}
            className="text-white/20 hover:text-red-400/70 transition-colors duration-150 text-lg leading-none cursor-pointer"
            aria-label="Remove example"
          >
            ×
          </button>
        )}
      </div>

      {/* Review textarea */}
      <textarea
        value={example.review}
        onChange={handleReviewChange}
        placeholder="Enter example review text..."
        rows={3}
        className="w-full bg-white/3 border border-white/8 focus:border-amber-500/30 rounded-lg px-4 py-3 text-white/80 placeholder:text-white/15 text-sm leading-relaxed resize-none focus:outline-none transition-colors duration-200 font-light"
      />

      {/* Star rating */}
      <div className="flex items-center gap-3">
        <span className="text-[11px] text-white/30 tracking-widest uppercase font-mono whitespace-nowrap">
          Rating
        </span>
        <div className="flex items-center gap-1">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              onClick={() => handleRatingChange(star)}
              className={`text-xl transition-all duration-100 cursor-pointer hover:scale-110 ${
                star <= example.rating
                  ? "text-amber-400"
                  : "text-white/15 hover:text-white/30"
              }`}
              aria-label={`Rate ${star} stars`}
            >
              ★
            </button>
          ))}
        </div>
        <span className="text-[11px] text-white/25 font-mono">
          {example.rating}/5
        </span>
      </div>
    </div>
  );
}
