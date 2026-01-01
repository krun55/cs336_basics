from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any

@dataclass
class BPETokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

def train_bpe(input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str], **kwargs):
    raise NotImplementedError
