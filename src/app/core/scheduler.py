from __future__ import annotations
"""Simple scheduler placeholder (batching, multiple symbols, future Redis/queue integration)."""
from typing import List


def chunk_symbols(symbols: List[str], size: int) -> List[List[str]]:
    return [symbols[i:i+size] for i in range(0, len(symbols), size)]
