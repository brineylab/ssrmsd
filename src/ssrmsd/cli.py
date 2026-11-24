from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from ssrmsd.format import write_pairwise, write_results
from ssrmsd.io import discover_batch_dirs, load_manifest, parse_labeled_path
from ssrmsd.score import score_batch, score_ensemble
from ssrmsd.types import EnsembleResult, ScoringOptions, SsrmsdError


def _parse_chain_list(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    chains = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not chains:
        raise SsrmsdError("Chain list option must contain at least one chain ID")
    return chains


def _add_shared_options(parser: argparse.ArgumentParser, *, default_workers: int) -> None:
    parser.add_argument(
        "--align-chains",
        default=None,
        help="Comma-separated reference chain IDs used for Kabsch superposition.",
    )
    parser.add_argument(
        "--score-chains",
        default=None,
        help="Comma-separated reference chain IDs over which RMSD is measured.",
    )
    parser.add_argument(
        "--ref-source",
        default=None,
        help="Source label providing reference chain IDs (strict: errors if absent).",
    )
    parser.add_argument(
        "--glob",
        dest="glob_pattern",
        default=None,
        help="Override structure discovery inside directories/archives with a glob pattern.",
    )
    parser.add_argument(
        "--include-top-level",
        action="store_true",
        help="Include top-level AF3 *_model.cif files alongside seed-*_sample-* outputs.",
    )
    parser.add_argument(
        "--split-models",
        action="store_true",
        help="Treat each MODEL block in a multi-model file as a separate ensemble member.",
    )
    parser.add_argument(
        "--min-chain-id",
        type=float,
        default=0.90,
        help="Minimum overlap sequence identity for chain matching (default: 0.90).",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.80,
        help="Minimum aligned CA coverage relative to max chain length (default: 0.80).",
    )
    parser.add_argument(
        "--linker-min-len",
        type=int,
        default=10,
        help="Minimum poly-G run length to split linked chains (0 disables; default: 10).",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=("csv", "tsv", "jsonl"),
        default="csv",
        help="Output format (default: csv).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path (defaults to stdout).",
    )
    parser.add_argument(
        "--pairwise-out",
        type=Path,
        default=None,
        help="Optional path to write all unordered pairwise RMSD rows.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Worker processes for batch execution (default: {default_workers}).",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort immediately if any ensemble fails.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the `ssrmsd` CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ssrmsd",
        description="Compute normalized and legacy ssRMSD for predicted structure ensembles.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    score_p = sub.add_parser("score", help="Score a single ensemble.")
    score_p.add_argument(
        "inputs",
        nargs="+",
        help="Structure files, directories, or archives (optionally prefixed with label=).",
    )
    score_p.add_argument(
        "--id",
        dest="ensemble_id",
        default=None,
        help="Ensemble identifier for output rows.",
    )
    _add_shared_options(score_p, default_workers=1)

    batch_p = sub.add_parser("batch", help="Score a batch of ensembles.")
    src_group = batch_p.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--dir",
        dest="dirs",
        action="append",
        help="Batch root directory (repeatable; optionally prefixed with label=).",
    )
    src_group.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV/TSV manifest with columns ensemble_id,path[,source].",
    )
    batch_p.add_argument(
        "--join",
        choices=("union", "intersection"),
        default="union",
        help="How to combine ensemble IDs across multiple --dir roots (default: union).",
    )
    batch_p.add_argument(
        "--delimiter",
        default=None,
        help="Explicit manifest delimiter (inferred from .tsv/.csv by default).",
    )
    _add_shared_options(batch_p, default_workers=os.cpu_count() or 1)

    return parser


def _build_options(args: argparse.Namespace) -> ScoringOptions:
    return ScoringOptions(
        align_chains=_parse_chain_list(args.align_chains),
        score_chains=_parse_chain_list(args.score_chains),
        ref_source=args.ref_source,
        glob_pattern=args.glob_pattern,
        include_top_level=args.include_top_level,
        split_models=args.split_models,
        min_chain_id=args.min_chain_id,
        min_coverage=args.min_coverage,
        linker_min_len=args.linker_min_len,
    )


def _emit_outputs(
    results: Sequence[EnsembleResult],
    *,
    output: Path | None,
    pairwise_out: Path | None,
    fmt: Literal["csv", "tsv", "jsonl"],
) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as handle:
            write_results(results, handle, fmt=fmt)
    else:
        write_results(results, sys.stdout, fmt=fmt)

    if pairwise_out is not None:
        pairwise_out.parent.mkdir(parents=True, exist_ok=True)
        with pairwise_out.open("w", encoding="utf-8", newline="") as handle:
            write_pairwise(results, handle, fmt=fmt)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    include_pairwise = args.pairwise_out is not None

    try:
        options = _build_options(args)
        if args.command == "score":
            res = score_ensemble(
                args.inputs,
                ensemble_id=args.ensemble_id,
                options=options,
                include_pairwise=include_pairwise,
            )
            _emit_outputs(
                [res],
                output=args.output,
                pairwise_out=args.pairwise_out,
                fmt=args.fmt,
            )
            return 0 if res.status == "ok" else 1

        if args.command == "batch":
            if args.manifest is not None:
                specs = load_manifest(args.manifest, delimiter=args.delimiter)
            else:
                roots = [parse_labeled_path(raw) for raw in args.dirs]
                specs = discover_batch_dirs(roots, join=args.join)
            if not specs:
                print("ssrmsd: no ensembles discovered", file=sys.stderr)
                return 1
            results = score_batch(
                specs,
                options=options,
                workers=args.workers,
                include_pairwise=include_pairwise,
                fail_fast=args.fail_fast,
            )
            _emit_outputs(
                results,
                output=args.output,
                pairwise_out=args.pairwise_out,
                fmt=args.fmt,
            )
            return 0

        parser.error(f"Unknown command: {args.command}")
        return 2
    except SsrmsdError as exc:
        print(f"ssrmsd: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
