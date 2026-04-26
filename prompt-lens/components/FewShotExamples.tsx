"use client";

import { Example } from "@/app/page";
import ExampleCard from "./ExampleCard";

interface FewShotExamplesProps {
  examples: Example[];
  onChange: (examples: Example[]) => void;
}

const MAX_EXAMPLES = 3;

function generateId(): string {
  return Math.random().toString(36).slice(2, 8);
}

export default function FewShotExamples({
  examples,
  onChange,
}: FewShotExamplesProps) {
  const handleUpdate = (id: string, updated: Example) => {
    onChange(examples.map((ex) => (ex.id === id ? updated : ex)));
  };

  const handleRemove = (id: string) => {
    onChange(examples.filter((ex) => ex.id !== id));
  };

  const handleAdd = () => {
    if (examples.length >= MAX_EXAMPLES) return;
    onChange([...examples, { id: generateId(), review: "", rating: 0 }]);
  };

  return (
    <div className="flex flex-col gap-4 animate-in fade-in slide-in-from-top-2 duration-300">
      <div className="flex items-center justify-between">
        <label className="text-[11px] text-white/30 tracking-widest uppercase font-mono">
          Few-shot Examples
        </label>
        <span className="text-[11px] text-white/20 font-mono">
          {examples.length}/{MAX_EXAMPLES}
        </span>
      </div>

      <div className="flex flex-col gap-3">
        {examples.map((example, index) => (
          <ExampleCard
            key={example.id}
            example={example}
            index={index}
            onChange={(updated) => handleUpdate(example.id, updated)}
            onRemove={() => handleRemove(example.id)}
            canRemove={examples.length > 1}
          />
        ))}
      </div>

      {examples.length < MAX_EXAMPLES && (
        <button
          onClick={handleAdd}
          className="w-full py-3 rounded-xl border border-dashed border-white/10 hover:border-amber-500/30 text-white/25 hover:text-amber-400/60 text-sm font-light transition-all duration-200 cursor-pointer group"
        >
          <span className="group-hover:scale-110 inline-block transition-transform duration-150 mr-1">
            +
          </span>
          Add Example
        </button>
      )}
    </div>
  );
}
