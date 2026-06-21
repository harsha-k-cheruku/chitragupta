from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Cost log lives next to the package root, shared across all pipelines
_LOG_PATH = Path(__file__).resolve().parent.parent / "costs.jsonl"

# Anthropic pricing per million tokens (update as models change)
_API_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6":          {"input": 3.00,  "output": 15.00},
    "claude-opus-4-7":            {"input": 15.00, "output": 75.00},
    "claude-haiku-4-5-20251001":  {"input": 0.80,  "output": 4.00},
}


def _cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = _API_PRICING.get(model)
    if not pricing:
        return 0.0
    return (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000


def log(
    project: str,
    section: str,
    task: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    entry = {
        "ts":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project":       project,
        "section":       section,
        "task":          task,
        "model":         model,
        "input_tokens":  input_tokens,
        "output_tokens": output_tokens,
        "cost_usd":      round(_cost_usd(model, input_tokens, output_tokens), 6),
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def summary(days: int = 7) -> None:
    """Print a spend summary for the last N days."""
    if not _LOG_PATH.exists():
        print("No cost log found.")
        return

    from datetime import timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    rows: list[dict] = []
    with _LOG_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if datetime.fromisoformat(r["ts"]) >= cutoff:
                    rows.append(r)
            except Exception:
                continue

    if not rows:
        print(f"No entries in the last {days} days.")
        return

    total = sum(r["cost_usd"] for r in rows)
    by_project: dict[str, float] = {}
    by_model:   dict[str, int]   = {}
    for r in rows:
        by_project[r["project"]] = by_project.get(r["project"], 0) + r["cost_usd"]
        by_model[r["model"]]     = by_model.get(r["model"], 0) + 1

    print(f"\n── HKC AI Cost Summary (last {days} days) ──────────────")
    print(f"  Total spend:  ${total:.4f}")
    print(f"  Total calls:  {len(rows)}")
    print(f"\n  By project:")
    for proj, cost in sorted(by_project.items(), key=lambda x: -x[1]):
        print(f"    {proj:<20} ${cost:.4f}")
    print(f"\n  By model:")
    for model, count in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"    {model:<40} {count} calls")
    print()
