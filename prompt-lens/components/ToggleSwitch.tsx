"use client";

import { Mode } from "@/app/page";

interface ToggleSwitchProps {
  mode: Mode;
  onChange: (mode: Mode) => void;
}

const options: { label: string; value: Mode }[] = [
  { label: "Zero-shot", value: "zero-shot" },
  { label: "Few-shot", value: "few-shot" },
];

export default function ToggleSwitch({ mode, onChange }: ToggleSwitchProps) {
  return (
    <div className="relative flex items-center bg-white/5 border border-white/10 rounded-full p-1 gap-1">
      {/* Sliding pill */}
      <div
        className={`absolute top-1 h-[calc(100%-8px)] w-[calc(50%-4px)] rounded-full bg-amber-500/90 shadow-lg shadow-amber-500/20 transition-all duration-300 ease-in-out ${
          mode === "few-shot" ? "left-[calc(50%+2px)]" : "left-1"
        }`}
      />

      {options.map(({ label, value }) => (
        <button
          key={value}
          onClick={() => onChange(value)}
          className={`relative z-10 px-6 py-2 rounded-full text-sm font-medium tracking-wide transition-colors duration-200 cursor-pointer ${
            mode === value
              ? "text-black"
              : "text-white/50 hover:text-white/80"
          }`}
        >
          {label}
        </button>
      ))}
    </div>
  );
}
