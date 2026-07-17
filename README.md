# Domain-Adapted LLM via LoRA Fine-Tuning

> Fine-tuned Llama 3 8B Instruct on 18,000 domain-specific instruction-response pairs using LoRA and QLoRA (4-bit quantization). Built for a policy and compliance bot requiring structured JSON output, business terminology alignment, and reduced hallucination on domain-specific queries.

---

## The Problem

Large language models like Llama 3 8B Instruct are strong general reasoners. But in enterprise deployments, general reasoning is not enough. The model:

- Did not know business-specific terminology
- Could not reliably produce structured JSON output without heavy prompt engineering
- Hallucinated answers to domain-specific questions where it lacked knowledge
- Required extensive prompt crafting to follow company-specific output formats consistently

The goal was not to change the model's reasoning ability — it already had that. The goal was **domain adaptation**: teaching the model to answer in the right format, using the right terminology, grounded in the right knowledge base.

Full fine-tuning of an 8B parameter model was not cost-effective for this use case. LoRA (Low-Rank Adaptation) allowed us to adapt the model's behaviour by training a small number of additional parameters while keeping the base weights frozen  achieving the domain alignment we needed at a fraction of the compute cost.

---

## Approach: LoRA vs Full Fine-Tuning

```
Full Fine-Tuning
├── Updates all ~8 billion parameters
├── Requires large GPU memory (80GB+)
├── High compute cost
├── Risk of catastrophic forgetting
└── Overkill for domain adaptation

LoRA (Low-Rank Adaptation)
├── Injects trainable rank-decomposition matrices
│   into attention layers only
├── Trains ~0.1-1% of total parameters
├── Runs on A100 40GB with 4-bit quantization
├── Preserves base model reasoning ability
└── Right-sized for domain alignment tasks
```

For a use case focused on terminology alignment, structured output formatting, and domain knowledge consistency rather than teaching new reasoning capabilities — LoRA is the correct choice. Full fine-tuning would have been more expensive and more likely to degrade the model's general instruction-following ability.

---

## Architecture

```
Base Model
Llama 3 8B Instruct
        │
        ▼
┌─────────────────────────────────────┐
│         QLoRA Configuration         │
│                                     │
│  4-bit quantization (bitsandbytes)  │
│  NF4 quantization type              │
│  Double quantization enabled        │
│  Compute dtype: bfloat16            │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         LoRA Adapter Config         │
│                                     │
│  Target modules: q_proj, v_proj,    │
│  k_proj, o_proj, gate_proj,         │
│  up_proj, down_proj                 │
│  Rank (r): 16                       │
│  Alpha: 32                          │
│  Dropout: 0.05                      │
│  Bias: none                         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│         Training Pipeline           │
│                                     │
│  Dataset (18,000 pairs)             │
│         │                           │
│         ▼                           │
│  Tokenization                       │
│  (max_length: 2048, padding left)   │
│         │                           │
│         ▼                           │
│  SFTTrainer (TRL)                   │
│  - Supervised fine-tuning           │
│  - Packing enabled                  │
│  - Gradient checkpointing           │
│         │                           │
│         ▼                           │
│  Validation loop                    │
│  (every N steps)                    │
│         │                           │
│         ▼                           │
│  Adapter merge                      │
│  (LoRA weights → base model)        │
│         │                           │
│         ▼                           │
│  Inference evaluation               │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Fine-Tuned Model               │
│      Llama 3 8B + LoRA Adapters     │
│                                     │
│  Deployed as internal policy bot    │
│  Structured JSON output             │
│  Domain-aligned responses           │
└─────────────────────────────────────┘
```

---

## Dataset

### Construction

18,000 supervised instruction-response pairs built from internal sources:

| Source | Contribution |
|---|---|
| Internal documentation | Core knowledge base |
| FAQs | Common query patterns |
| Annotated examples | High-quality reference outputs |
| Historical support conversations | Real-world query distribution |

### Preprocessing Pipeline

```
Raw data sources
      │
      ▼
Deduplication
(exact + near-duplicate removal)
      │
      ▼
Format standardisation
(consistent instruction-response template)
      │
      ▼
Anonymisation
(PII removal, sensitive data scrubbing)
      │
      ▼
Quality filtering
(minimum response length, format validation)
      │
      ▼
Instruction template conversion
      │
      ▼
Train / Validation / Test split
80% / 10% / 10%
(14,400 / 1,800 / 1,800 pairs)
```

### Instruction Template

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a domain expert assistant. Answer questions accurately
using company terminology. Always respond in valid JSON format.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}
<|eot_id|>
```

---

## Training Configuration

### Hardware

| Environment | Use |
|---|---|
| NVIDIA A100 40GB | Production training runs |
| Google Colab Pro (A100) | Quick experiments and ablations |

4-bit quantization via QLoRA reduced peak GPU memory from ~32GB (full precision) to ~12GB, making A100 40GB sufficient for the full training run.

### Key Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| LoRA rank (r) | 16 | Balance between capacity and parameter efficiency |
| LoRA alpha | 32 | Alpha/r ratio of 2 for stable training |
| LoRA dropout | 0.05 | Light regularisation |
| Learning rate | 2e-4 | Standard for LoRA fine-tuning |
| Batch size | 4 (effective 16 with grad accum) | Memory constraint on A100 40GB |
| Gradient accumulation | 4 steps | Simulate larger batch |
| Max sequence length | 2048 tokens | Covers longest instruction-response pairs |
| Epochs | 3 | Validated against overfitting on held-out set |
| Warmup ratio | 0.03 | Stable loss from start |
| LR scheduler | Cosine | Smooth decay for fine-tuning |
| Quantisation | 4-bit NF4 (bitsandbytes) | Memory efficiency |
| Compute dtype | bfloat16 | Numerical stability |


| Library | Version | Role |
|---|---|---|
| transformers | 4.40+ | Model loading, tokenisation, inference |
| peft | 0.10+ | LoRA adapter configuration and training |
| trl | 0.8+ | SFTTrainer for supervised fine-tuning |
| bitsandbytes | 0.43+ | 4-bit quantisation (QLoRA) |
| datasets | 2.18+ | Dataset loading and preprocessing |
| accelerate | 0.29+ | Distributed training and mixed precision |

---

## Evaluation

### Quantitative Metrics

| Metric | Base Model | Fine-Tuned | Delta |
|---|---|---|---|
| Validation Loss | — | Converged by epoch 2 | — |
| Perplexity | Higher | Lower on domain corpus | Improved |
| Exact Match (JSON outputs) | ~45% | ~81% | +36pp |
| ROUGE-L (long-form answers) | Baseline | +12% | Improved |

### Human Evaluation

300 held-out examples manually reviewed and scored across four dimensions:

| Dimension | Base Model | Fine-Tuned | Notes |
|---|---|---|---|
| Factual correctness | Moderate | High | Fewer domain errors |
| Instruction adherence | Inconsistent | Consistent | Format followed reliably |
| Output formatting | ~45% valid JSON | ~89% valid JSON | Major improvement |
| Hallucination rate | High on domain queries | Significantly reduced | Key goal achieved |

**Why human evaluation mattered more than NLP benchmarks:**

This was a business application. ROUGE and BLEU scores measure surface similarity to reference outputs useful for translation and summarisation. For a policy bot where correctness, format compliance, and hallucination rate are the actual success criteria, human evaluation against real business queries is more meaningful.

### What the Evaluation Caught

**Epoch selection:** Training for more than 3 epochs caused the validation loss to start rising (overfitting to training format). Epoch 3 checkpoint was selected as the production model.

**JSON reliability:** The base model produced valid JSON approximately 45% of the time under standard prompting. Post fine-tuning, this rose to approximately 89% without any additional prompt engineering. The remaining ~11% failures were on edge cases with unusually long or nested output structures.

**Hallucination pattern:** The base model most commonly hallucinated on queries involving specific internal policies and terminology that did not appear in its pretraining data. Post fine-tuning, hallucination on these query types was significantly reduced because the training data directly covered them.

---

## Key Design Decisions

**LoRA over full fine-tuning**
Domain adaptation, not new capability acquisition. Training ~0.5% of parameters via LoRA achieved the alignment we needed without the cost or risk of full fine-tuning. Catastrophic forgetting of general instruction-following ability was a real concern with full fine-tuning on an 18K dataset.

**QLoRA for memory efficiency**
4-bit NF4 quantisation reduced peak GPU memory from ~32GB to ~12GB. This made A100 40GB sufficient for production training and enabled faster iteration during development experiments on Colab.

**SFTTrainer over custom training loop**
TRL's SFTTrainer handles packing, padding, and the instruction masking (masking loss on the instruction tokens so the model only learns to predict the response) out of the box. Writing a custom training loop for supervised fine-tuning adds complexity without benefit.

**Human evaluation as primary metric**
For a policy bot in a business context, exact match and ROUGE scores are misleading. A response can be factually correct, well-formatted, and appropriately concise while scoring poorly on ROUGE if the wording differs from the reference. Human evaluation against real business queries is the ground truth.

**Adapter merge for inference**
After training, LoRA adapters were merged back into the base model weights for inference. This eliminates the adapter overhead at inference time and simplifies deployment the merged model behaves identically to a normally fine-tuned model at serving time.

---

## Results

| Metric | Result |
|---|---|
| Training dataset | 18,000 instruction-response pairs |
| Base model | Llama 3 8B Instruct |
| Fine-tuning method | QLoRA (4-bit) + LoRA adapters |
| JSON output reliability | 45% (base) → 89% (fine-tuned) |
| Domain hallucination | Significantly reduced |
| Human eval: instruction adherence | Consistent vs inconsistent (base) |
| Training hardware | NVIDIA A100 40GB |
| Prompt engineering required post fine-tune | Significantly less |

---



**Human evaluation is non-negotiable for business applications.** Automated metrics would have told a misleading story. The real improvement fewer domain errors, consistent formatting, reduced hallucination on company-specific queries only shows up clearly in human evaluation against real business queries.
