"""Command-line entry point for the single Leonos experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json
from .config import DEFAULT_CONFIG, load_config


def _audit_data(config: dict[str, Any]) -> dict[str, Any]:
    from .data import (
        apply_quality_policy,
        audit_daily_panel,
        exchange_sessions,
        fetch_daily_snapshot,
        load_daily_panel,
        load_manifest,
        normalize_daily_bars,
        to_canonical_bars,
        write_audit_reports,
    )

    raw_root = Path(config["paths"]["raw_data"])
    source = config["sources"]["dataset"]
    manifest_path = fetch_daily_snapshot(
        raw_root,
        revision=str(source["revision"]),
        repo_id=str(source["repo_id"]),
    )
    raw = load_daily_panel(raw_root, manifest_path=manifest_path)
    normalized, normalization = normalize_daily_bars(raw)
    canonical = to_canonical_bars(normalized)
    calendar = exchange_sessions(canonical["session"].min(), canonical["session"].max())
    summary_root = Path(config["paths"]["summaries"])
    summary_root.mkdir(parents=True, exist_ok=True)
    raw_audit = audit_daily_panel(normalized, calendar_sessions=calendar)
    write_audit_reports(
        raw_audit,
        summary_root / "data_raw_audit.json",
        summary_root / "data_raw_audit.md",
    )
    accepted, exclusions = apply_quality_policy(canonical, calendar)
    accepted_source = accepted.drop(columns=["ticker", "session"])
    accepted_audit = audit_daily_panel(accepted_source, calendar_sessions=calendar)
    write_audit_reports(
        accepted_audit,
        summary_root / "data_acceptance.json",
        summary_root / "data_acceptance.md",
    )
    atomic_write_json(summary_root / "data_manifest.json", load_manifest(manifest_path))
    atomic_write_json(
        summary_root / "data_policy.json",
        {
            "normalization": normalization,
            "raw_rows": len(raw),
            "accepted_rows": len(accepted),
            "exclusion_rows": len(exclusions),
            "exclusions_by_reason": exclusions["reason"].value_counts().to_dict(),
            "raw_audit_accepted": raw_audit["acceptance"]["accepted"],
            "accepted_audit_accepted": accepted_audit["acceptance"]["accepted"],
        },
    )
    if not accepted_audit["acceptance"]["accepted"]:
        raise RuntimeError("post-policy daily data audit failed; see results/summary")
    return {
        "manifest": str(manifest_path),
        "raw_rows": len(raw),
        "accepted_rows": len(accepted),
        "excluded_rows": len(exclusions),
        "ticker_count": accepted["ticker"].nunique(),
        "accepted": True,
    }


def _predict_lightgbm(config: dict[str, Any], *, split: str, seed: int) -> Path:
    # The baseline command already persists both validation and test predictions.
    # Re-entering that provenance gate avoids a second path that could combine a
    # stale prepared panel, model, or configuration under a canonical filename.
    from .pipeline import fit_baseline

    completed = fit_baseline(config, seed=seed)
    return Path(completed[f"{split}_predictions"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="leonos", description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="pinned daily data operations")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    data_commands.add_parser("fetch", help="download/reuse the pinned daily snapshot")
    data_commands.add_parser("audit", help="run raw and post-policy acceptance audits")
    commands.add_parser("prepare", help="create accepted bars, targets, and cached features")

    baseline = commands.add_parser("baseline", help="LightGBM operations")
    baseline_commands = baseline.add_subparsers(dest="baseline_command", required=True)
    fit = baseline_commands.add_parser("fit", help="validation search and declared final refit")
    fit.add_argument("--seed", type=int, default=42)

    smoke = commands.add_parser("smoke", help="small real-data Kronos validation smoke")
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--batch-size", type=int, default=2)
    smoke.add_argument("--limit", type=int, default=2)
    smoke.add_argument("--seed", type=int, default=42)

    benchmark = commands.add_parser(
        "benchmark-kronos", help="brief post-smoke validation batch sweep"
    )
    benchmark.add_argument("--device", default="cuda:0")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8, 16, 32, 64])

    predict = commands.add_parser("predict", help="persist model predictions")
    predict.add_argument("--model", choices=("kronos", "lightgbm"), required=True)
    predict.add_argument("--split", choices=("validation", "test"), required=True)
    predict.add_argument("--seed", type=int, default=42)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--batch-size", type=int)
    predict.add_argument("--shard-index", type=int, default=0)
    predict.add_argument("--num-shards", type=int)

    evaluate = commands.add_parser("evaluate", help="evaluate saved predictions only")
    evaluate.add_argument("--seed", type=int, default=42)
    scenario = commands.add_parser(
        "scenario",
        help="CPU-only paired block bootstrap of saved RankIC and portfolio returns",
    )
    scenario.add_argument("--replicates", type=int)
    scenario.add_argument("--block-length", type=int)
    scenario.add_argument("--seed", type=int)
    scenario.add_argument("--scenario-config", default="configs/scenario.yaml")
    commands.add_parser("report", help="render the result report from saved metrics")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "data" and args.data_command == "fetch":
        from .data import fetch_daily_snapshot

        source = config["sources"]["dataset"]
        result: Any = fetch_daily_snapshot(
            config["paths"]["raw_data"],
            revision=str(source["revision"]),
            repo_id=str(source["repo_id"]),
        )
    elif args.command == "data" and args.data_command == "audit":
        result = _audit_data(config)
    elif args.command == "prepare":
        from .pipeline import prepare_data

        result = prepare_data(config)
    elif args.command == "baseline" and args.baseline_command == "fit":
        from .pipeline import fit_baseline

        result = fit_baseline(config, seed=args.seed)
    elif args.command == "smoke":
        from .kronos_runner import run_kronos_predictions

        result = run_kronos_predictions(
            config,
            split="validation",
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            limit=args.limit,
            run_name="validation-smoke",
        )
    elif args.command == "benchmark-kronos":
        from .kronos_runner import benchmark_kronos_batches

        result = benchmark_kronos_batches(
            config,
            device=args.device,
            seed=args.seed,
            batch_sizes=args.batch_sizes,
        )
    elif args.command == "predict" and args.model == "lightgbm":
        result = _predict_lightgbm(config, split=args.split, seed=args.seed)
    elif args.command == "predict" and args.model == "kronos":
        from .kronos_runner import frozen_kronos_execution_plan, run_kronos_predictions

        batch_size, num_shards = frozen_kronos_execution_plan(
            config, batch_size=args.batch_size, num_shards=args.num_shards
        )
        result = run_kronos_predictions(
            config,
            split=args.split,
            seed=args.seed,
            device=args.device,
            batch_size=int(batch_size),
            shard_index=args.shard_index,
            num_shards=num_shards,
        )
    elif args.command == "evaluate":
        from .reporting import evaluate_saved_predictions

        result = evaluate_saved_predictions(config, seed=args.seed)
    elif args.command == "scenario":
        from .scenario import load_scenario_config, run_saved_scenario_analysis

        result = run_saved_scenario_analysis(
            config,
            scenario_config=load_scenario_config(args.scenario_config),
            replicates=args.replicates,
            block_length=args.block_length,
            seed=args.seed,
        )
    elif args.command == "report":
        from .figures import render_result_figures
        from .reporting import render_results_report

        report_path = render_results_report(config)
        result = {"report": report_path, "figures": render_result_figures(config)}
    else:  # pragma: no cover - argparse makes this unreachable
        raise RuntimeError("unhandled command")
    if isinstance(result, Path):
        print(result)
    else:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
