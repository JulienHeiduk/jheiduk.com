---
title: "Fine-Tune LLMs with QLoRA"
date: 2026-02-24
draft: false
summary: "Fine-tune a large language model with QLoRA on a single GPU, then serve it at high throughput using vLLM's PagedAttention engine."
---

Fine-tuning a large language model on commodity hardware used to require painful trade-offs. **QLoRA** removed the memory wall by combining 4-bit quantization with low-rank adapters. **vLLM** removes the inference wall by rethinking how GPU memory is managed during serving. Together they form a practical, end-to-end pipeline — from experiment to production — that fits on a single A100 or even a Colab T4.

This article walks through the complete workflow: adapt a causal LM with QLoRA, then load the result into vLLM for batched, high-throughput generation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1OxQhcAYBWyPSO8Ac4pSIDZFWk-8DwQpt?usp=sharing)

> **Note on embedding:** Google Colab sets `X-Frame-Options: SAMEORIGIN` on all its pages, which prevents browsers from rendering the notebook inside an `<iframe>`. The badge above is the standard workaround — it opens the notebook in a new Colab tab with a single click.

## 1. Why vLLM?

Standard HuggingFace generation allocates a fixed KV-cache block per sequence upfront. Because sequence lengths vary, most of that memory sits unused. At batch size 32 with mixed-length prompts, 60–80% of reserved memory is wasted.

vLLM introduces **PagedAttention**: the KV cache is split into fixed-size pages (like OS virtual memory), allocated on demand and freed immediately when a sequence finishes. The gains are concrete:

- Near-zero internal memory fragmentation
- **Continuous batching** — new requests slot in mid-batch instead of waiting for the current batch to drain
- 20–30× higher throughput than naive `model.generate()` at the same latency
- Native support for LoRA adapters, GPTQ/AWQ quantization, and tensor parallelism

## 2. Fine-Tuning with QLoRA

**LoRA** (Low-Rank Adaptation) freezes the base model weights and injects small trainable matrices $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ into each target layer:

$$\Delta W = B \cdot A, \quad r \ll \min(d, k)$$

With rank $r = 16$, a 7B model goes from ~7B trainable parameters to ~4M — a 99.9% reduction.

**QLoRA** stacks 4-bit NF4 quantization on top:
- Base model weights stored in 4-bit (NF4 format, preserving the normal distribution of weights)
- LoRA adapters computed and stored in bf16
- **Double quantization** compresses the quantization constants themselves, saving ~0.5 GB per 7B model
- **Paged optimizers** offload Adam states to CPU RAM, preventing OOM spikes during gradient steps

The result: fine-tuning a 7B model in under 12 GB of VRAM.

## 3. The Fine-Tuning Pipeline

### Requirements

```bash
uv pip install transformers peft trl bitsandbytes accelerate datasets vllm
```

### Step 1 — Load the base model in 4-bit

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # swap for Mistral-7B, Llama-3-8B, etc.

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NF4 is optimal for normally distributed weights
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,       # quantize the quantization constants
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # TinyLlama has no dedicated pad token
```

### Step 2 — Attach LoRA adapters

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)  # enables gradient checkpointing for 4-bit

lora_config = LoraConfig(
    r=16,                                              # rank — higher = more capacity
    lora_alpha=32,                                     # scaling: effective lr ∝ alpha/r
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # attention projections
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 2,883,584 || all params: 1,103,224,832 || trainable%: 0.26
```

### Step 3 — Train with SFTTrainer

```python
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 2,000 Python instruction-following examples in Alpaca format
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")

# Combine instruction / input / output into a single text field
dataset = dataset.map(lambda x: {
    "text": f"### Instruction:\n{x['instruction']}\n\n### Input:\n{x['input']}\n\n### Response:\n{x['output']}"
})
dataset = dataset.remove_columns(["prompt"])  # drop original prompt column

sft_config = SFTConfig(
    output_dir="./tinyllama-python-adapter",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch = 16
    warmup_steps=50,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    dataset_text_field="text",
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()
trainer.model.save_pretrained("./tinyllama-python-adapter")
tokenizer.save_pretrained("./tinyllama-python-adapter")
```

## 4. Models comparison

Let's try the original model and fine-tuned model.

**Step 1 — Merge the adapter into base weights:**

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base in full precision and fuse LoRA weights (ΔW = BA)
base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
merged = PeftModel.from_pretrained(base, "./tinyllama-python-adapter")
merged = merged.merge_and_unload()   # fuses ΔW = BA into the base weights
merged.save_pretrained("./tinyllama-merged")
tokenizer.save_pretrained("./tinyllama-merged")
print("Merged model saved.")
```

**Step 2 — Before/after comparison with the HuggingFace pipeline:**

```python
from transformers import pipeline

# Original model — no fine-tuning
pipe = pipeline("text-generation", model=MODEL_ID, torch_dtype="auto", device_map="auto")
output = pipe("Write a Python function that reverses a linked list.", max_new_tokens=200)
print(output[0]["generated_text"])
```

```python
# Fine-tuned model — adapter merged into weights
pipe = pipeline("text-generation", model="./tinyllama-merged", torch_dtype="auto", device_map="auto")
output = pipe("Write a Python function that reverses a linked list.", max_new_tokens=200)
print(output[0]["generated_text"])
```

The base model rephrases the prompt. The fine-tuned version returns actual Python code following the Alpaca response format it was trained on.

## Conclusion

QLoRA makes fine-tuning accessible on a single consumer GPU by combining 4-bit NF4 quantization with low-rank adapters. vLLM handles the serving side, replacing naïve `model.generate()` with PagedAttention and continuous batching.

The two libraries are complementary by design: train with the PEFT/TRL ecosystem, serve with vLLM. For multi-adapter deployments, vLLM's dynamic LoRA loading means you only need one copy of the base weights in GPU memory, regardless of how many specialized adapters you accumulate.

**References:**
- [vLLM documentation](https://docs.vllm.ai/)
- [QLoRA paper — "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [PEFT library (HuggingFace)](https://huggingface.co/docs/peft/)
- [TRL — Transformer Reinforcement Learning](https://huggingface.co/docs/trl/)

The companion notebook with all the code from this article is available [on GitHub](https://github.com/JulienHeiduk/jheiduk.com/tree/main/notebooks/vllm_qlora_finetuning.ipynb).
