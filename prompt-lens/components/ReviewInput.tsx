"use client";

import { useRef } from "react";

interface ReviewInputProps {
  value: string;
  onChange: (value: string) => void;
}

export default function ReviewInput({ value, onChange }: ReviewInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    onChange(e.target.value);
    // Auto-resize
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  };

  return (
    <div className="flex flex-col gap-2">
      <label className="text-[11px] text-white/30 tracking-widest uppercase font-mono">
        Movie Review
      </label>

      <div className="relative group">
        {/* Subtle border highlight on focus (no glow flood) */}
        <div className="absolute -inset-px rounded-xl border border-transparent transition-colors duration-200 pointer-events-none" />

        <textarea
          ref={textareaRef}
          value={value}
          onChange={handleChange}
          placeholder="Enter your movie review..."
          rows={6}
          className="w-full bg-white/3 border border-white/10 rounded-xl px-5 py-4 text-white/90 placeholder:text-white/20 text-sm leading-relaxed resize-none focus:outline-none focus:border-amber-500/30 transition-colors duration-200 font-light min-h-40"
        />

        {/* Char count */}
        <div className="absolute bottom-3 right-4 text-[10px] text-white/20 font-mono tabular-nums pointer-events-none">
          {value.length}
        </div>
      </div>
    </div>
  );
}
