from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ssrmsd.score import score_batch, score_ensemble
from ssrmsd.types import (
    ChainData,
    EnsembleResult,
    EnsembleSpec,
    PairwiseDistance,
    ScoringOptions,
    SsrmsdError,
    StructureData,
    StructureInput,
)

try:
    __version__: str = version("ssRMSD")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "ChainData",
    "EnsembleResult",
    "EnsembleSpec",
    "PairwiseDistance",
    "ScoringOptions",
    "SsrmsdError",
    "StructureData",
    "StructureInput",
    "__version__",
    "score_batch",
    "score_ensemble",
]
