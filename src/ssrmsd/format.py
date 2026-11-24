from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from typing import Any, Literal, TextIO

from ssrmsd.types import EnsembleResult

SUMMARY_COLUMNS: tuple[str, ...] = (
    "ensemble_id",
    "n_structures",
    "n_ca_align",
    "n_ca_score",
    "mean_sq_rmsd",
    "rms_rmsd",
    "mean_rmsd",
    "median_rmsd",
    "rmsd_sd",
    "ssRMSD",
    "SD",
    "within_source_mean_rmsd",
    "cross_source_mean_rmsd",
    "sources",
    "status",
    "error",
)

PAIRWISE_COLUMNS: tuple[str, ...] = (
    "ensemble_id",
    "struct_i",
    "struct_j",
    "source_i",
    "source_j",
    "rmsd",
)

_FLOAT_FIELDS = frozenset(
    {
        "mean_sq_rmsd",
        "rms_rmsd",
        "mean_rmsd",
        "median_rmsd",
        "rmsd_sd",
        "ssRMSD",
        "SD",
        "within_source_mean_rmsd",
        "cross_source_mean_rmsd",
    }
)


def _round_opt(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _result_to_dict(result: EnsembleResult, *, for_tabular: bool) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for col in SUMMARY_COLUMNS:
        val = getattr(result, col)
        if col in _FLOAT_FIELDS:
            val = _round_opt(val)
        if for_tabular and val is None:
            row[col] = ""
        else:
            row[col] = val
    return row


def write_results(
    results: Sequence[EnsembleResult],
    stream: TextIO,
    *,
    fmt: Literal["csv", "tsv", "jsonl"] = "csv",
) -> None:
    """Write summary `EnsembleResult` rows to `stream` in CSV, TSV, or JSONL format."""
    if fmt == "jsonl":
        for res in results:
            stream.write(json.dumps(_result_to_dict(res, for_tabular=False)) + "\n")
        return

    delimiter = "\t" if fmt == "tsv" else ","
    writer = csv.DictWriter(
        stream, fieldnames=SUMMARY_COLUMNS, delimiter=delimiter, lineterminator="\n"
    )
    writer.writeheader()
    for res in results:
        writer.writerow(_result_to_dict(res, for_tabular=True))


def write_pairwise(
    results: Sequence[EnsembleResult],
    stream: TextIO,
    *,
    fmt: Literal["csv", "tsv", "jsonl"] = "csv",
) -> None:
    """Write unordered pairwise distance rows across `results` to `stream`."""
    if fmt == "jsonl":
        for res in results:
            for pair in res.pairwise:
                rec = {
                    "ensemble_id": pair.ensemble_id,
                    "struct_i": pair.struct_i,
                    "struct_j": pair.struct_j,
                    "source_i": pair.source_i,
                    "source_j": pair.source_j,
                    "rmsd": round(pair.rmsd, 4),
                }
                stream.write(json.dumps(rec) + "\n")
        return

    delimiter = "\t" if fmt == "tsv" else ","
    writer = csv.DictWriter(
        stream, fieldnames=PAIRWISE_COLUMNS, delimiter=delimiter, lineterminator="\n"
    )
    writer.writeheader()
    for res in results:
        for pair in res.pairwise:
            writer.writerow(
                {
                    "ensemble_id": pair.ensemble_id,
                    "struct_i": pair.struct_i,
                    "struct_j": pair.struct_j,
                    "source_i": "" if pair.source_i is None else pair.source_i,
                    "source_j": "" if pair.source_j is None else pair.source_j,
                    "rmsd": round(pair.rmsd, 4),
                }
            )
