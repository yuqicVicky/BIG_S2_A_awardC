"""Shared LLM utilities — single model constant and two call helpers."""

from __future__ import annotations

import json
import re

MODEL = "claude-sonnet-4-6"


def call_llm_text(client, prompt: str, max_tokens: int = 400) -> str:
    """Call Claude and return plain text. Returns empty string on any failure."""
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in resp.content if b.type == "text"), "")
    except Exception:
        return ""


def call_llm_json(client, prompt: str, max_tokens: int = 600) -> dict | list | None:
    """Call Claude and parse JSON from the response. Returns None on any failure."""
    full_prompt = prompt + "\n\nReturn ONLY valid JSON (no markdown fences, no extra text)."
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": full_prompt}],
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        text = re.sub(r"```[a-z]*\n?", "", text).strip().rstrip("`")
        m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return None
