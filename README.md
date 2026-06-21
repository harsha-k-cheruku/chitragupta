# Chitragupta

> *In Hindu mythology, Chitragupta is the cosmic accountant — the impartial clerk who classifies every soul's deeds, routes them to the right destination, and keeps flawless records. This is that, for AI model calls.*

A lightweight model router for Python AI pipelines. Route tasks to a local model (Ollama) or a cloud API (Anthropic Claude) based on what the task actually requires — not because it's the only option you wired up. Log every call and its cost automatically.

---

## The Problem

Most AI pipelines send every call to the most expensive frontier model, regardless of whether the task needs it. Summarising a document, extracting structured data, or classifying text doesn't require GPT-4 or Claude Sonnet. A well-tuned 14B local model handles these tasks adequately — for free.

The tasks that genuinely need frontier models are fewer: multi-step reasoning, synthesis across conflicting sources, creative generation, or iterative self-challenge loops. Chitragupta makes this distinction explicit and automatic.

---

## How It Works

Every call is tagged with a **task type**. Task types map to a model tier. The router handles the rest — including format translation between providers.

```
extraction    → local    (pull structured data from documents)
summarization → local    (condense text into structured output)
classification→ local    (rate, label, categorise)
reasoning     → api      (synthesis, cross-domain judgment)
creative      → api      (narrative, dialogue, long-form writing)
socratic      → api      (multi-turn self-challenge loops)
```

You can override the tier for any task type via environment variables or extend the mapping in code.

---

## Installation

```bash
pip install -e /path/to/chitragupta
```

Requires Python 3.11+. Dependencies: `anthropic`, `openai` (used as the Ollama client).

**Local model:** Install [Ollama](https://ollama.com) and pull a model:
```bash
brew install ollama
ollama serve
ollama pull qwen2.5:14b    # recommended: good quality, fits in 16GB RAM
```

**Cloud API:** Set your Anthropic key:
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

### Single-turn call

```python
from hkc_ai import call_model

result = call_model(
    system="You are a risk analyst. Extract the top material risks.",
    user_parts=[{"type": "text", "text": filing_text}],
    task="extraction",        # → routes to local model
    project="my_pipeline",   # for cost logging
    section="risk_extraction",
)
```

### Multi-turn call (e.g. iterative reasoning loops)

```python
from hkc_ai import call_model_messages

messages = [
    {"role": "user", "content": "Analyse this company's competitive position..."}
]
# Add assistant response, then continue the conversation
result = call_model_messages(
    messages=messages,
    system="You are a rigorous analyst...",
    task="socratic",          # → always routes to API
    project="my_pipeline",
    section="deep_analysis",
    max_tokens=2000,
)
```

### Cost summary

```python
from hkc_ai import cost_log

cost_log.summary(days=7)
```

```
── Chitragupta Cost Summary (last 7 days) ──────────
  Total spend:  $0.1633
  Total calls:  13

  By project:
    my_pipeline          $0.1633

  By model:
    claude-sonnet-4-6    8 calls
    qwen2.5:14b          5 calls
```

---

## Configuration

Override defaults via environment variables:

```bash
export HKC_AI_API_MODEL=claude-sonnet-4-6        # default cloud model
export HKC_AI_LOCAL_MODEL=qwen2.5:14b            # default local model
export HKC_AI_LOCAL_URL=http://localhost:11434/v1 # Ollama endpoint
```

---

## Fallback Behaviour

If Ollama is unavailable (not running, model not pulled), local calls automatically fall back to the cloud API with a warning — the pipeline keeps running.

```
⚠ Ollama unavailable (ConnectionError), falling back to API for risk_extraction
```

---

## Extending

**Add a new task type:**
```python
from hkc_ai.router import TASK_TIERS
TASK_TIERS["translation"] = "local"
```

**Add a new provider:** Implement `_call_<provider>()` in `router.py` following the same signature as `_call_local()` and `_call_api()`, then add it as a tier option.

---

## Cost Log Format

Every call appends one line to `costs.jsonl` (in the package root):

```json
{"ts": "2026-06-21T18:31:41+00:00", "project": "my_pipeline", "section": "risk_extraction", "task": "extraction", "model": "qwen2.5:14b", "input_tokens": 1611, "output_tokens": 297, "cost_usd": 0.0}
{"ts": "2026-06-21T18:38:38+00:00", "project": "my_pipeline", "section": "thesis", "task": "reasoning", "model": "claude-sonnet-4-6", "input_tokens": 1002, "output_tokens": 1159, "cost_usd": 0.020391}
```

Local model calls are logged with `cost_usd: 0.0`. This gives you a complete picture of which tasks genuinely needed the cloud.

---

## Recommended Local Models

| Model | RAM required | Best for |
|---|---|---|
| `qwen2.5:7b` | ~6GB | Fast, good at structured tasks |
| `qwen2.5:14b` | ~10GB | Best quality/cost ratio — recommended |
| `llama3.3:70b` | ~40GB | Approaches frontier quality |
| `deepseek-r1:8b` | ~6GB | Reasoning tasks on constrained hardware |

On Apple Silicon (M1/M2/M3/M4), Ollama uses the unified memory pool efficiently — a 14B model runs well on a 16GB MacBook.

---

## Why Not LiteLLM?

LiteLLM is a great library for teams that need to support 100+ providers, proxy servers, and enterprise observability. Chitragupta is for developers who want a 150-line, auditable router they can read in five minutes — with no transitive dependencies beyond the two provider SDKs they already use. Smaller attack surface, simpler mental model, easier to extend.

---

*Not affiliated with Anthropic or Ollama. Built for personal AI pipelines.*
