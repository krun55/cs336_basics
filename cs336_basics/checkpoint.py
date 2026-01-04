import torch
from typing import IO, BinaryIO
import os

def save_checkpoint(model, optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]):
    raise NotImplementedError

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model, optimizer) -> int:
    raise NotImplementedError
