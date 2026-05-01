# 🎬 PromptLens: Comparative Study of SLMs and LLMs for Movie Review Rating

## 📌 Overview

PromptLens is a comparative AI system designed to analyze movie reviews and predict a rating on a scale of 1 to 5. The project explores the behavior and performance differences between Small Language Models (SLMs) and Large Language Models (LLMs), with a strong focus on **zero-shot and few-shot prompting techniques**.

The system allows users to input a movie review, choose between zero-shot and few-shot modes, and observe how different models interpret the same input. It then compares predictions with user-provided ground truth and evaluates model performance using metrics like accuracy and mean absolute error.

---

## 🎯 Objectives

- Implement a Small Language Model (SLM) for classification tasks
- Experiment with **zero-shot and few-shot prompting**
- Analyze how prompting strategies influence model outputs
- Compare SLM outputs with stronger LLM outputs
- Evaluate model performance using standard metrics

---

## 🧠 Models Used

The system evaluates four different models:

### 🔹 1. Custom Encoder Model (SLM)
- Built and trained on movie review datasets
- Acts as the baseline small language model
- Provides structured, learned predictions

### 🔹 2. Small LLM (Qwen 0.5B)
- Lightweight language model
- Fine-tuned on movie review data
- Enhanced using prompt engineering techniques

### 🔹 3. Large LLM (llama-3.3-70b-versatile)
- Accessed via OpenRouter / Groq
- No fine-tuning applied
- Uses carefully designed prompts for prediction
- Serves as a high-capability reference model

---

## 🔬 Fine-tuning Approach Comparison

| | Custom Encoder (BERT) | Small LLM (Qwen 0.5B) |
|---|---|---|
| **Base architecture** | BERT-base (encoder, built from scratch) | Qwen2.5-0.5B (decoder, pretrained) |
| **Training method** | Full fine-tune | LoRA (PEFT) — adapters only |
| **Trainable parameters** | ~110M (all) | ~10M (LoRA r=8 + classifier head) |
| **Checkpoint size** | ~450 MB | ~10–30 MB |
| **Dataset** | Amazon Reviews 2023 — Books (1.5M samples) | Same |
| **Dataset cache** | Created and pushed to HF Hub | Reused — no re-download |
| **HF Hub repo** | `AgentPhoenix7/SLM-project` | `AgentPhoenix7/SLM-project-qwen` |
| **Kaggle notebook** | `bert-finetune-kaggle.ipynb` | `qwen-finetune-kaggle.ipynb` |

---

## ⚙️ Prompting Techniques

### 🔸 Zero-shot Prompting
- No examples provided
- Model relies solely on instruction and internal knowledge

### 🔸 Few-shot Prompting
- Up to 3 examples are provided
- Examples influence the model's prediction behavior
- Demonstrates how contextual guidance affects output

### 🔸 Prompt Optimization
- Structured prompts for consistent output
- Controlled formatting (strict 1–5 rating output)
- Experimental variations to improve reliability

---

## 🖥️ System Workflow

1. User enters a movie review
2. Selects prompting mode (Zero-shot / Few-shot)
3. (Optional) Provides few-shot examples
4. System processes input through all four models
5. Each model outputs a rating (1–5)
6. User provides actual rating
7. System evaluates predictions and displays performance metrics

---

## 📊 Evaluation Metrics

- **Accuracy**
- **Mean Absolute Error (MAE)**
- Model agreement comparison

---

## 🧩 Tech Stack

- **Frontend:** Next.js, React, Tailwind CSS
- **Backend:** Node.js API routes / server functions
- **Model APIs:** OpenRouter, Groq
- **ML Libraries:** Scikit-learn (for encoder models)

---

## 🧪 Features

- Real-time model comparison (2x2 grid view)
- Toggle between zero-shot and few-shot prompting
- Dynamic few-shot example input
- Structured output visualization
- Performance evaluation dashboard

---

## 👥 Team Contributions

- **Biprarshi**
  - Development of custom encoder model
  - Training and dataset handling

- **Anis & David**
  - Fine-tuning of small LLM (Qwen 0.5B)
  - Prompt engineering for small LLM

- **Arin**
  - Frontend development
  - Backend API integration
  - Prompt engineering for large LLM
  - System orchestration and evaluation pipeline

---