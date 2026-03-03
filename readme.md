# 🦙 Medical Q&A Fine-Tuning with LoRA + TinyLlama-1.1B

> Parameter-efficient fine-tuning of a 1.1B LLM — training only **0.5% of weights** while preserving full general capability.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sonal12061/lora-tinyllama-medical-qa/blob/main/LoRA_TinyLlama.ipynb)

---

## What This Is

Fine-tuning TinyLlama-1.1B on USMLE-style medical MCQs using LoRA (Low-Rank Adaptation) — a parameter-efficient method that injects small trainable adapter matrices into frozen attention layers.

The goal was not to build a medical AI. The goal was to answer a specific question:

> *Can you teach a general-purpose LLM to follow a structured MCQ format — consistently, concisely, without hallucination — by training less than 1% of its weights?*

The answer is yes. And the more interesting finding was in the evaluation.

---

## How LoRA Works

For a weight matrix **W** in an attention layer (e.g. query projection, 2048×2048 = 4M params):

Instead of computing: `W_new = W + ΔW` (4M parameter update)

LoRA computes: `W_new = W + B × A × (alpha/r)` where:
- **A**: `[r × d]` — 8 × 2048 = 16K params
- **B**: `[d × r]` — 2048 × 8 = 16K params
- **Total per layer**: 32K instead of 4M — a **125× reduction**

Base weights are completely frozen. Only A and B receive gradients.

---

## Results

### What Changed After Fine-Tuning

| | Base Model | Fine-Tuned (LoRA) |
|---|---|---|
| Output format | Verbose paragraph | Crisp `A: answer` |
| Follows Alpaca template | ❌ | ✅ |
| Stays in MCQ format | ❌ | ✅ |
| Hallucinates answer letters | ✅ Yes | ❌ No |
| Response length | 20–30 words | 3–5 words |

### The Catastrophic Forgetting Test

After medical fine-tuning, I tested the model on a completely out-of-domain prompt:

```
Prompt: Write a SQL query to find the top 5 highest paid employees.

Fine-tuned model: SELECT * FROM employees ORDER BY salary DESC LIMIT 5;
```

Correct. General capability fully preserved — because frozen weights were never touched.

### Why No Accuracy Number?

For a **generative model**, measuring accuracy on 3 questions has zero statistical power. The real proof of instruction fine-tuning is **output behaviour**, not letter scores on a handful of questions.

I also caught a silent bug in my own evaluator — the base model response said *"C: Ciprofloxacin"* when Ciprofloxacin is option A. My `extract_letter` function grabbed the hallucinated "C" and marked it correct. After patching the evaluator with disease-name matching, the base model's real score dropped from 3/3 to 2/3.

The full story is in the blog below.

---

## Key Design Decisions

**Why `r=8`?**
Rank controls the bottleneck dimension of adapter matrices. r=8 means adapters represent task adaptation as 8 underlying directions. For instruction fine-tuning — teaching format and structure — 8 directions is sufficient. The original LoRA paper validated r=4 to r=8 for most NLP tasks.

**Why `lora_alpha=16`?**
Effective scale = `alpha/r = 16/8 = 2.0`. This gives the adapter enough voice to steer behaviour without overwhelming pre-trained representations. Empirically, `alpha = 2×r` works well for instruction fine-tuning.

**Why only `q_proj` and `v_proj`?**
Query determines what the model attends to (task-dependent). Value determines what information flows forward (task-dependent). Applying LoRA to Q and V achieves results competitive with all four attention matrices at half the parameter count.

**Why `lr=2e-4` (10× higher than DistilBERT fine-tuning)?**
Catastrophic forgetting risk is zero when base weights are frozen. The adapter matrices start from zero and need a higher LR to learn. In full fine-tuning, a high LR destroys pre-trained representations immediately — here it can't.

**Why `labels = input_ids`?**
This is causal language modelling — not classification. At every token position, the model predicts the next token. Setting `labels = input_ids` computes loss across all 512 positions simultaneously, teaching the model the full Alpaca template structure and the correct answer format after `### Response:`.

---

## Pipeline

```
TinyLlama-1.1B (frozen base weights — 1.1B params)
        ↓
Inject LoRA Adapters into q_proj + v_proj
Trainable params: ~6M (0.5% of total)
        ↓
Load MedAlpaca medical_meadow_medqa (USMLE questions)
        ↓
Format using Alpaca prompt template
### Instruction / ### Input / ### Response
        ↓
Tokenize: max_length=512, labels=input_ids
        ↓
Train: 3 epochs, lr=2e-4, batch=4, grad_accum=4
        ↓
Save adapter only (~few MB vs ~2GB full model)
        ↓
Compare: Fine-Tuned vs Base TinyLlama-1.1B-Chat
```

---

## How to Run

**Google Colab (recommended):** Click the badge above — T4 GPU, everything pre-configured.

**Locally:**
```bash
git clone https://github.com/Sonal12061/lora-tinyllama-medical-qa
cd lora-tinyllama-medical-qa
pip install transformers peft datasets accelerate bitsandbytes torch
jupyter notebook LoRA_TinyLlama_Formatted.ipynb
```

---

## Stack

`HuggingFace PEFT` · `HuggingFace Transformers` · `TinyLlama-1.1B` · `MedAlpaca` · `PyTorch` · `Google Colab T4`

---

## How This Fits Into the Portfolio

This repo sits directly after full fine-tuning in the learning progression:

```
ml-from-scratch               ← understand model internals
        ↓
distilbert-sentiment-analysis ← full fine-tuning, all weights updated
        ↓
lora-tinyllama-medical-qa     ← parameter-efficient fine-tuning (this repo)
        ↓
LangGraph_Implementations     ← orchestrate multiple LLM agents
```

The companion piece is the DistilBERT repo — same task family (supervised NLP), opposite fine-tuning strategy. Together they answer the question: *when do you fine-tune everything, and when do you use adapters?*

🔗 [DistilBERT Full Fine-Tuning](https://github.com/Sonal12061/distilbert-sentiment-analysis)

---

## ✍️ Deep Dive Blog

The full story — including the evaluator bug, the measurement problem, and why accuracy is the wrong metric for generative models:

📖 [I Fine-Tuned a 1.1B LLM on Medical Questions Using LoRA — Then Spent More Time Questioning My Evaluation Than Writing the Training Code](https://medium.com/@sonal.mishra1297)

---

## References

- [LoRA Paper — Hu et al. 2021](https://arxiv.org/abs/2106.09685)
- [HuggingFace PEFT Documentation](https://huggingface.co/docs/peft)
- [TinyLlama Model Card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- [MedAlpaca Dataset](https://huggingface.co/datasets/medalpaca/medical_meadow_medqa)