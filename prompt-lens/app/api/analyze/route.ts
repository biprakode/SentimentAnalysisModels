import { NextRequest, NextResponse } from "next/server";
import { generateText } from "ai";
import { groq } from '@ai-sdk/groq';

import { Example } from "@/app/page";

const delay = (ms: number) => new Promise((res) => setTimeout(res, ms));

// Custom Encoder
async function customEncoder() {
    await delay(1000);
    return 4;
}

// HuggingFace Encoder
async function hfEncoder() {
    await delay(1000);
    return 4;
}

// Qwen Small LLM
async function qwen_zs() {
    await delay(1000);
    return 4;
}

async function qwen_fs() {
    await delay(1000);
    return 3;
}

// Groq LLM

async function groq_zs(review : string) {

    const SYSTEM_PROMPT_ZS = `
        You are a strict E-commerce product review rating system.

        Your task is to analyze a E-commerce product review and assign a rating from 1 to 5.

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
        const rating = match ? parseInt(match[0]) : 0;  //zero rating as fallback
        
        return rating

    } catch (error: any) {
        return Response.json({
            success: false,
            message: "Invalid request. Unable to perform analysis.",
        }, { status: 400 } );
    }
}

async function groq_fs(review: string, examples: Example[]): Promise<any> {

    const SYSTEM_PROMPT_FS = `
        You are a strict E-commerce product review rating system.

        Your task is to analyze a E-commerce product review and assign a rating from 1 to 5.

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
        return Response.json({
            success: false,
            message: "Invalid request. Unable to perform analysis.",
        }, { status: 400 } );
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

    const [custom, hf, qwen, groq] = await Promise.all([
      customEncoder(),
      hfEncoder(),
      isFewShot ? qwen_fs() : qwen_zs(),
      isFewShot ? groq_fs(review, examples) : groq_zs(review),
    ]);

    const totalTime = Date.now() - start;

    // Return the response
    return NextResponse.json({
      meta: {
        mode,
        reviewLength: review.length,
        examplesCount: examples.length,
        latency: totalTime,
      },
      results: [
        { model: "custom", rating: custom },
        { model: "hf", rating: hf },
        { model: "qwen", rating: qwen },
        { model: "groq", rating: groq },
      ],
    });
  } catch (err) {
    console.error("API Error:", err);

    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}