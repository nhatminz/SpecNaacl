"""Resolve historical FastLK helpers first, then shared SpecNaacl utilities."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[3] / "helper"))
