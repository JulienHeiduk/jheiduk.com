---
title: "Fine-Tune LLMs with QLoRA and Serve Them with vLLM"
date: 2026-02-24
draft: false
summary: "Fine-tune a large language model with QLoRA on a single GPU, then serve it at high throughput using vLLM's PagedAttention engine."
---

Fine-tuning a large language model on commodity hardware used to require painful trade-offs. **QLoRA** removed the memory wall by combining 4-bit quantization with low-rank adapters. **vLLM** removes the inference wall by rethinking how GPU memory is managed during serving. Together they form a practical, end-to-end pipeline — from experiment to production — that fits on a single A100 or even a Colab T4.

This article walks through the complete workflow: adapt a causal LM with QLoRA, then load the result into vLLM for batched, high-throughput generation.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JulienHeiduk/jheiduk.com/blob/main/notebooks/vllm-qlora-finetuning.ipynb)

> **Note on embedding:** Google Colab sets `X-Frame-Options: SAMEORIGIN` on all its pages, which prevents browsers from rendering the notebook inside an `<iframe>`. The badge above is the standard workaround — it opens the notebook in a new Colab tab with a single click.

## 1. Why vLLM?

Standard HuggingFace generation allocates a fixed KV-cache block per sequence upfront. Because sequence lengths vary, most of that memory sits unused. At batch size 32 with mixed-length prompts, 60–80% of reserved memory is wasted.

vLLM introduces **PagedAttention**: the KV cache is split into fixed-size pages (like OS virtual memory), allocated on demand and freed immediately when a sequence finishes. The gains are concrete:

- Near-zero internal memory fragmentation
- **Continuous batching** — new requests slot in mid-batch instead of waiting for the current batch to drain
- 20–30× higher throughput than naive `model.generate()` at the same latency
- Native support for LoRA adapters, GPTQ/AWQ quantization, and tensor parallelism

<!-- Diagram: Show vLLM's PagedAttention KV-cache as paged blocks vs. contiguous pre-allocated blocks in standard inference, side by side. Include labels for fragmentation waste in the standard approach and zero-waste in PagedAttention. -->
![vllm-paged-attention](/vllm-paged-attention.svg)
*Figure: PagedAttention allocates KV-cache in pages, eliminating fragmentation.*

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
pip install transformers peft trl bitsandbytes accelerate datasets vllm
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

# 2,000 Python instruction-following examples from Alpaca format
dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train[:2000]")

sft_config = SFTConfig(
    output_dir="./tinyllama-python-adapter",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch = 16
    warmup_steps=50,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=50,
    max_seq_length=512,
    dataset_text_field="output",
)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()
trainer.model.save_pretrained("./tinyllama-python-adapter")
tokenizer.save_pretrained("./tinyllama-python-adapter")
```

## 4. Serving with vLLM

vLLM supports two patterns for fine-tuned models.

**Option A — Merge adapter into base weights** (recommended for production with a single adapter):

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base in full precision, merge LoRA weights, save
base = AutoModelForCausalLM.from_pretrained(MODEL_ID)
merged = PeftModel.from_pretrained(base, "./tinyllama-python-adapter")
merged = merged.merge_and_unload()   # fuses ΔW = BA into the base weights
merged.save_pretrained("./tinyllama-merged")

from vllm import LLM, SamplingParams

llm = LLM(model="./tinyllama-merged")
outputs = llm.generate(
    ["Write a Python function that reverses a linked list."],
    SamplingParams(temperature=0.7, max_tokens=256),
)
print(outputs[0].outputs[0].text)
```

**Option B — Dynamic LoRA loading** (for serving multiple adapters from a single base model):

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# One base model instance, many adapters loaded on demand
llm = LLM(model=MODEL_ID, enable_lora=True, max_lora_rank=64)

outputs = llm.generate(
    ["Write a Python function that reverses a linked list."],
    SamplingParams(temperature=0.7, max_tokens=256),
    lora_request=LoRARequest(
        lora_name="python_adapter",
        lora_int_id=1,
        lora_local_path="./tinyllama-python-adapter",
    ),
)
print(outputs[0].outputs[0].text)
```

Option B is the pattern to reach for when you have a fleet of task-specific adapters (SQL generation, code review, summarization) and want to route requests without the overhead of maintaining separate model replicas.

## Conclusion

QLoRA makes fine-tuning accessible on a single consumer GPU by combining 4-bit NF4 quantization with low-rank adapters. vLLM handles the serving side, replacing naïve `model.generate()` with PagedAttention and continuous batching.

The two libraries are complementary by design: train with the PEFT/TRL ecosystem, serve with vLLM. For multi-adapter deployments, vLLM's dynamic LoRA loading means you only need one copy of the base weights in GPU memory, regardless of how many specialized adapters you accumulate.

**References:**
- [vLLM documentation](https://docs.vllm.ai/)
- [QLoRA paper — "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)
- [PEFT library (HuggingFace)](https://huggingface.co/docs/peft/)
- [TRL — Transformer Reinforcement Learning](https://huggingface.co/docs/trl/)
