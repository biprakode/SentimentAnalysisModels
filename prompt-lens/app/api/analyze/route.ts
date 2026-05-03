import { NextRequest, NextResponse } from "next/server";
import { generateText } from "ai";
import { groq } from '@ai-sdk/groq';

import { Example } from "@/app/page";

// Custom Encoder — calls bert_server.py
const BERT_API_URL = process.env.BERT_API_URL ?? "http://localhost:8001";

async function custom_zs(review: string): Promise<number> {
    const res = await fetch(`${BERT_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review, mode: "zero-shot" }),
    });
    if (!res.ok) throw new Error(`BERT server error: ${res.status}`);
    const { rating } = await res.json();
    return rating;
}

async function custom_fs(review: string, examples: Example[]): Promise<number> {
    const res = await fetch(`${BERT_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review, mode: "few-shot", examples: examples.slice(0, 3) }),
    });
    if (!res.ok) throw new Error(`BERT server error: ${res.status}`);
    const { rating } = await res.json();
    return rating;
}

// HuggingFace Encoder (disabled)
// async function hf_zs() {
//     await delay(1000);
//     return 4;
// }

// async function hf_fs() {
//     await delay(1000);
//     return 3;
// }

// Qwen Small LLM — calls the local Python inference server (qwen_server.py)
const QWEN_API_URL = process.env.QWEN_API_URL ?? "http://localhost:8000";

async function qwen_zs(review: string): Promise<number> {
    const res = await fetch(`${QWEN_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review, mode: "zero-shot", optimize: true }),
    });
    if (!res.ok) throw new Error(`Qwen server error: ${res.status}`);
    const { rating } = await res.json();
    return rating;
}

async function qwen_fs(review: string, examples: Example[]): Promise<number> {
    const res = await fetch(`${QWEN_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ review, mode: "few-shot", examples: examples.slice(0, 3), optimize: true }),
    });
    if (!res.ok) throw new Error(`Qwen server error: ${res.status}`);
    const { rating } = await res.json();
    return rating;
}

// Groq LLM

async function groq_zs(review : string) {

    const SYSTEM_PROMPT_ZS = `
        You are a strict Amazon book review rating system.

        Your task is to analyze an Amazon book review and assign a rating from 1 to 5.

        Rating scale:
        1 = Very Negative
        2 = Negative
        3 = Neutral
        4 = Positive
        5 = Very Positive

        Rules:
        - Output ONLY a single number (1, 2, 3, 4, or 5)
        - Do NOT explain your answer
        - Do NOT output any text other than the number
        - Be consistent and conservative in scoring
        - Avoid extreme ratings unless strongly justified

        Example outputs:
        1
        3
        5
    `;

    try {        
        const result = await generateText({
            model: groq("llama-3.3-70b-versatile"),
            maxOutputTokens: 20,
            messages: [
                {
                    role: "system",
                    content: SYSTEM_PROMPT_ZS,
                },
                {
                    role: "user",
                    content: `Review:\n"${review}"\n\nRating (1-5):`
                },
            ]
        });

        const text = result.text?.trim() || "";

        const match = text.match(/[1-5]/);
        const rating = match ? parseInt(match[0]) : 3;

        return rating;

    } catch (error: any) {
        console.error("Groq (zero-shot) error:", error);
        return 3;
    }
}

async function groq_fs(review: string, examples: Example[]): Promise<any> {

    const SYSTEM_PROMPT_FS = `
        You are a strict Amazon book review rating system.

        Your task is to analyze an Amazon book review and assign a rating from 1 to 5.

        Rating scale:
        1 = Very Negative
        2 = Negative
        3 = Neutral
        4 = Positive
        5 = Very Positive

        IMPORTANT:
        If examples are provided, you MUST learn from them and follow their rating behavior strictly.

        This means:
        - Do NOT rely only on general sentiment understanding
        - Adapt your rating style based on the examples
        - If the examples show that positive reviews receive low ratings, you MUST follow that pattern
        - If the examples are inconsistent with typical sentiment, prioritize the examples over your own judgment
        - Treat the examples as the ground truth for how ratings should be assigned

        However:
        - If the examples are inconsistent, or appear arbitrary (no clear pattern between review sentiment and rating),
        then fall back to standard sentiment-based reasoning
        - In such cases, ignore unreliable examples and use your own judgment

        Rules:
        - Output ONLY a single number (1, 2, 3, 4, or 5)
        - Do NOT explain your answer
        - Do NOT output any text other than the number
        - Be consistent with the examples provided
        - If no examples are provided or examples are inconsistent, use standard sentiment-based reasoning
    `;

    try {
        const messages: any[] = [
        {
            role: "system",
            content: SYSTEM_PROMPT_FS,
        },
        ];

        // Add few-shot examples
        examples.slice(0, 3).forEach((ex) => {
        messages.push(
            {
            role: "user",
            content: `Review:\n"${ex.review}"\n\nRating (1-5):`,
            },
            {
            role: "assistant",
            content: `${ex.rating}`,
            }
        );
        });

        // Add actual input
        messages.push({
            role: "user",
            content: `Review:\n"${review}"\n\nRating (1-5):`,
        });

        const result = await generateText({
            model: groq("llama-3.3-70b-versatile"),
            maxOutputTokens: 20,
            messages,
        });

        const text = result.text?.trim() || "";

        const match = text.match(/[1-5]/);
        const rating = match ? parseInt(match[0]) : 3;

        return rating;

    } catch (error) {
        console.error("Groq (few-shot) error:", error);
        return 3;
    }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();

    const {
      review,
      mode, // "zero-shot" | "few-shot"
      examples = [],
    } = body;

    // Basic validation
    if (!review || !mode) {
      return NextResponse.json(
        { error: "Missing review or mode" },
        { status: 400 }
      );
    }

    const isFewShot = mode === "few-shot" && examples.length > 0;

    // Call all models in parallel
    const start = Date.now();

    // const [custom, hf, qwen, groq] = await Promise.allSettled([
    //   isFewShot ? custom_fs(review, examples) : custom_zs(review),
    //   isFewShot ? hf_fs() : hf_zs(),
    //   isFewShot ? qwen_fs(review, examples) : qwen_zs(review),
    //   isFewShot ? groq_fs(review, examples) : groq_zs(review),
    // ]);
    const [customResult, qwenResult, groqResult] = await Promise.allSettled([
      isFewShot ? custom_fs(review, examples) : custom_zs(review),
      isFewShot ? qwen_fs(review, examples) : qwen_zs(review),
      isFewShot ? groq_fs(review, examples) : groq_zs(review),
    ]);

    const resolve = (r: PromiseSettledResult<number>, name: string) => {
      if (r.status === "fulfilled") return r.value;
      console.error(`${name} failed:`, r.reason);
      return null;
    };

    const custom = resolve(customResult, "BERT");
    const qwen   = resolve(qwenResult,   "Qwen");
    const groq   = resolve(groqResult,   "Groq");

    const totalTime = Date.now() - start;

    const allResults = [
      custom !== null ? { model: "custom", rating: custom } : null,
      // hf !== null ? { model: "hf", rating: hf } : null,
      qwen   !== null ? { model: "qwen",   rating: qwen }   : null,
      groq   !== null ? { model: "groq",   rating: groq }   : null,
    ].filter(Boolean);

    return NextResponse.json({
      meta: {
        mode,
        reviewLength: review.length,
        examplesCount: examples.length,
        latency: totalTime,
      },
      results: allResults,
    });
  } catch (err) {
    console.error("API Error:", err);

    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}