from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


class SsrmsdError(ValueError):
    """Domain error raised for invalid or unscoreable ensemble inputs."""


@dataclass(frozen=True)
class ChainData:
    """Extracted polymer chain sequence and C-alpha coordinates."""

    chain_id: str
    sequence: str
    ca_coords: np.ndarray  # shape (L, 3), dtype float64


@dataclass(frozen=True)
class StructureData:
    """Parsed structure containing extracted polymer chains."""

    name: str
    source: str | None
    chains: tuple[ChainData, ...]
    was_split: bool = False


@dataclass(frozen=True)
class StructureInput:
    """Path to a structure file, directory, or archive, with optional source tag."""

    path: Path
    source: str | None = None


@dataclass(frozen=True)
class ScoringOptions:
    """Configuration controlling structure discovery, alignment, and scoring."""

    align_chains: tuple[str, ...] | None = None
    score_chains: tuple[str, ...] | None = None
    ref_source: str | None = None
    glob_pattern: str | None = None
    include_top_level: bool = False
    split_models: bool = False
    min_chain_id: float = 0.90
    min_coverage: float = 0.80
    linker_min_len: int = 10


@dataclass(frozen=True)
class PairwiseDistance:
    """One unordered pairwise RMSD entry between two structures in an ensemble."""

    ensemble_id: str
    struct_i: str
    struct_j: str
    source_i: str | None
    source_j: str | None
    rmsd: float


@dataclass(frozen=True)
class EnsembleSpec:
    """Named ensemble composed of one or more structure inputs."""

    ensemble_id: str
    inputs: tuple[StructureInput, ...]


@dataclass(frozen=True)
class EnsembleResult:
    """Summary metrics and optional pairwise distances for one scored ensemble."""

    ensemble_id: str
    n_structures: int
    n_ca_align: int | None
    n_ca_score: int | None
    mean_sq_rmsd: float | None
    rms_rmsd: float | None
    mean_rmsd: float | None
    median_rmsd: float | None
    rmsd_sd: float | None
    ssRMSD: float | None
    SD: float | None
    within_source_mean_rmsd: float | None
    cross_source_mean_rmsd: float | None
    sources: str
    status: Literal["ok", "error"]
    error: str
    pairwise: tuple[PairwiseDistance, ...] = ()

    @classmethod
    def from_error(
        cls,
        ensemble_id: str,
        *,
        n_structures: int = 0,
        sources: str = "",
        error: str,
    ) -> EnsembleResult:
        """Build a failed ensemble result with empty metric fields."""
        return cls(
            ensemble_id=ensemble_id,
            n_structures=n_structures,
            n_ca_align=None,
            n_ca_score=None,
            mean_sq_rmsd=None,
            rms_rmsd=None,
            mean_rmsd=None,
            median_rmsd=None,
            rmsd_sd=None,
            ssRMSD=None,
            SD=None,
            within_source_mean_rmsd=None,
            cross_source_mean_rmsd=None,
            sources=sources,
            status="error",
            error=error,
            pairwise=(),
        )
