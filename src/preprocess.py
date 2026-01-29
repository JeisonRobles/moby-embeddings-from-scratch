from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


GUTENBERG_START_RE = re.compile(r"\*\*\*\s*START OF(.*?)\*\*\*", re.IGNORECASE)
GUTENBERG_END_RE = re.compile(r"\*\*\*\s*END OF(.*?)\*\*\*", re.IGNORECASE)


@dataclass
class PreprocessConfig:
    min_paragraph_chars: int = 120
    lowercase: bool = True


def strip_gutenberg_header_footer(text: str) -> str:
    """
    Remove Project Gutenberg boilerplate.
    """
    start = GUTENBERG_START_RE.search(text)
    end = GUTENBERG_END_RE.search(text)

    if start and end and start.end() < end.start():
        return text[start.end(): end.start()].strip()
    return text.strip()


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def split_paragraphs(text: str) -> List[str]:
    """
    Split text into paragraphs using blank lines.
    """
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


def basic_clean(text: str, lowercase: bool = True) -> str:
    if lowercase:
        text = text.lower()

    # Keep letters, numbers and apostrophes
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_paragraph_dataset(
    raw_text: str,
    cfg: PreprocessConfig
) -> List[Tuple[int, str, str]]:
    """
    Returns: [(paragraph_id, raw_paragraph, cleaned_paragraph)]
    """
    core = strip_gutenberg_header_footer(raw_text)
    core = normalize_whitespace(core)

    paragraphs = split_paragraphs(core)

    rows = []
    pid = 0
    for p in paragraphs:
        cleaned = basic_clean(p, lowercase=cfg.lowercase)
        if len(cleaned) < cfg.min_paragraph_chars:
            continue
        rows.append((pid, p, cleaned))
        pid += 1

    return rows
