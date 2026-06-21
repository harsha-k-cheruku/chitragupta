from __future__ import annotations

import os
from typing import Any

import anthropic
from openai import OpenAI

from hkc_ai import cost_log

# ── Task → model tier mapping ─────────────────────────────────
# Override by setting HKC_AI_LOCAL_MODEL or HKC_AI_API_MODEL env vars.
TASK_TIERS: dict[str, str] = {
    "extraction":    "local",   # structured parsing from documents
    "summarization": "local",   # condensing text into structured output
    "classification":"local",   # rating, labelling, categorising
    "reasoning":     "api",     # synthesis, cross-domain judgment
    "creative":      "api",     # narrative, dialogue, long-form writing
    "socratic":      "api",     # multi-turn self-challenge loop
}

_API_MODEL   = os.environ.get("HKC_AI_API_MODEL",   "claude-sonnet-4-6")
_LOCAL_MODEL = os.environ.get("HKC_AI_LOCAL_MODEL",  "qwen2.5:14b")
_LOCAL_URL   = os.environ.get("HKC_AI_LOCAL_URL",    "http://localhost:11434/v1")

_anthropic_client: anthropic.Anthropic | None = None
_ollama_client:    OpenAI | None = None


def _get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def _get_ollama() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(base_url=_LOCAL_URL, api_key="ollama")
    return _ollama_client


def _flatten_parts(user_parts: list[dict]) -> str:
    """Convert Anthropic-style content blocks to a plain string for Ollama."""
    return "\n\n".join(
        p["text"] for p in user_parts if p.get("type") == "text"
    )


# ── Single-turn call ──────────────────────────────────────────

def call_model(
    system: str,
    user_parts: list[dict],
    task: str,
    project: str = "",
    section: str = "",
    max_tokens: int = 2048,
) -> str:
    tier = TASK_TIERS.get(task, "api")

    if tier == "local":
        return _call_local(system, user_parts, project, section, task, max_tokens)
    return _call_api(system, user_parts, project, section, task, max_tokens)


def _call_api(
    system: str,
    user_parts: list[dict],
    project: str,
    section: str,
    task: str,
    max_tokens: int,
) -> str:
    resp = _get_anthropic().messages.create(
        model=_API_MODEL,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": user_parts}],
    )
    cost_log.log(
        project=project, section=section, task=task, model=_API_MODEL,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return resp.content[0].text.strip()


def _call_local(
    system: str,
    user_parts: list[dict],
    project: str,
    section: str,
    task: str,
    max_tokens: int,
) -> str:
    try:
        content = _flatten_parts(user_parts)
        resp = _get_ollama().chat.completions.create(
            model=_LOCAL_MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": content},
            ],
        )
        input_tokens  = getattr(resp.usage, "prompt_tokens",     0) or 0
        output_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
        cost_log.log(
            project=project, section=section, task=task, model=_LOCAL_MODEL,
            input_tokens=input_tokens, output_tokens=output_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠ Ollama unavailable ({e.__class__.__name__}), falling back to API for {section}")
        return _call_api(system, user_parts, project, section, task, max_tokens)


# ── Multi-turn call (Socratic loop — always API) ──────────────

def call_model_messages(
    messages: list[dict],
    system: str,
    task: str = "socratic",
    project: str = "",
    section: str = "",
    max_tokens: int = 1500,
) -> str:
    resp = _get_anthropic().messages.create(
        model=_API_MODEL,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=messages,
    )
    cost_log.log(
        project=project, section=section, task=task, model=_API_MODEL,
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
    )
    return resp.content[0].text.strip()
