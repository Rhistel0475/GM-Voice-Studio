"""
Utilities for safely parsing JSON emitted by LLMs.

These helpers stay conservative:
- accept normal JSON and fenced JSON
- ignore trailing prose after the first valid payload
- salvage complete entries from truncated top-level arrays
- remove trailing commas outside strings
"""
from __future__ import annotations

import json
from json import JSONDecodeError
from typing import Any


def strip_json_fences(raw_text: str) -> str:
    text = (raw_text or "").strip()
    if not text.startswith("```"):
        return text

    parts = text.split("```")
    if len(parts) < 2:
        return text

    text = parts[1]
    if text.startswith("json"):
        text = text[4:]
    return text.strip()


def _candidate_variants(raw_text: str) -> list[str]:
    text = strip_json_fences(raw_text)
    starts = [index for index in (text.find("["), text.find("{")) if index != -1]
    if starts:
        text = text[min(starts):]
    text = text.strip()
    if not text:
        return [text]

    without_trailing_commas = _remove_trailing_commas(text)
    if without_trailing_commas == text:
        return [text]
    return [text, without_trailing_commas]


def _remove_trailing_commas(text: str) -> str:
    output: list[str] = []
    in_string = False
    escaped = False
    index = 0

    while index < len(text):
        char = text[index]
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue

        if char == '"':
            in_string = True
            output.append(char)
            index += 1
            continue

        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "]}":
                index += 1
                continue

        output.append(char)
        index += 1

    return "".join(output)


def _raw_decode_first_value(text: str) -> Any:
    decoder = json.JSONDecoder()
    value, _end = decoder.raw_decode(text)
    return value


def _salvage_top_level_array(text: str) -> list[Any] | None:
    stripped = text.lstrip()
    if not stripped.startswith("["):
        return None

    decoder = json.JSONDecoder()
    index = 1
    items: list[Any] = []

    while index < len(stripped):
        while index < len(stripped) and stripped[index].isspace():
            index += 1
        if index >= len(stripped):
            break
        if stripped[index] == "]":
            return items
        if stripped[index] == ",":
            index += 1
            continue

        try:
            item, end = decoder.raw_decode(stripped, index)
        except JSONDecodeError:
            break

        items.append(item)
        index = end

    return items if items else None


def parse_llm_json(raw_text: str) -> Any:
    last_error: JSONDecodeError | None = None

    for candidate in _candidate_variants(raw_text):
        if not candidate:
            continue

        try:
            return json.loads(candidate)
        except JSONDecodeError as exc:
            last_error = exc

        try:
            return _raw_decode_first_value(candidate)
        except JSONDecodeError as exc:
            last_error = exc

    variants = _candidate_variants(raw_text)
    for candidate in variants:
        salvaged = _salvage_top_level_array(candidate)
        if salvaged is not None:
            return salvaged

    if last_error is not None:
        raise last_error
    raise JSONDecodeError("Empty JSON payload", raw_text or "", 0)


def parse_llm_json_array(raw_text: str) -> list[Any]:
    payload = parse_llm_json(raw_text)
    if not isinstance(payload, list):
        raise JSONDecodeError("Expected a JSON array", str(raw_text or ""), 0)
    return payload
