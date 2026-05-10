#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(REPO_ROOT))

from evaluation import run_batch_metrics as batch_metrics


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Recover evaluation result folders from a saved OpenAI/Azure Batch output JSONL."
  )
  parser.add_argument("input_dir", type=Path, help="Directory containing the original conversation JSON files.")
  parser.add_argument("output_dir", type=Path, help="Evaluation output directory to populate.")
  parser.add_argument(
    "--phase",
    default="panas_before",
    choices=("panas_before",),
    help="Batch phase to recover. Default: panas_before",
  )
  parser.add_argument(
    "--phase-dir",
    type=Path,
    default=None,
    help="Phase artifact directory containing output.jsonl and request_map.json.",
  )
  parser.add_argument(
    "--output-jsonl",
    type=Path,
    default=None,
    help="Raw batch output JSONL. Defaults to PHASE_DIR/output.jsonl.",
  )
  parser.add_argument(
    "--request-map",
    type=Path,
    default=None,
    help=(
      "Request map JSON. Defaults to PHASE_DIR/request_map.json. If omitted and unavailable, "
      "a PANAS-before map is reconstructed from input_dir and the custom IDs in output_jsonl."
    ),
  )
  parser.add_argument(
    "--max-iter",
    type=int,
    default=batch_metrics.DEFAULT_MAX_ITER,
    help=f"Expected evaluator samples per request. Default: {batch_metrics.DEFAULT_MAX_ITER}",
  )
  parser.add_argument(
    "--input-cost-per-1m",
    type=float,
    default=batch_metrics.DEFAULT_INPUT_COST_PER_1M,
    help="Input-token cost used to reconstruct cost fields.",
  )
  parser.add_argument(
    "--output-cost-per-1m",
    type=float,
    default=batch_metrics.DEFAULT_OUTPUT_COST_PER_1M,
    help="Output-token cost used to reconstruct cost fields.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite complete output files. Incomplete or missing files are always written.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Parse and report what would be written without creating result files.",
  )
  return parser.parse_args(argv)


def infer_output_jsonl(args: argparse.Namespace) -> Path:
  if args.output_jsonl is not None:
    return args.output_jsonl
  if args.phase_dir is not None:
    return args.phase_dir / "output.jsonl"
  raise batch_metrics.BatchMetricsError("Provide --output-jsonl or --phase-dir.")


def infer_request_map(args: argparse.Namespace) -> Path | None:
  if args.request_map is not None:
    return args.request_map
  if args.phase_dir is not None:
    candidate = args.phase_dir / "request_map.json"
    if candidate.is_file():
      return candidate
  return None


def reconstructed_panas_before_request_map(
  records: list[batch_metrics.ConversationRecord],
  output_jsonl: Path,
) -> dict[str, dict[str, Any]]:
  output_ids = set(batch_metrics.parse_batch_output_lines(output_jsonl))
  request_map: dict[str, dict[str, Any]] = {}
  for index, record in enumerate(records):
    custom_id = batch_metrics.make_custom_id(
      "panas_before",
      "panas_before",
      index,
      record.filename,
      "panas_before",
    )
    if custom_id not in output_ids:
      continue
    request_map[custom_id] = {
      "custom_id": custom_id,
      "phase": "panas_before",
      "metric_group": "panas_before",
      "input_file": record.filename,
      "output_file": (Path("panas_before") / record.filename).as_posix(),
      "item_key": "panas_before",
    }

  unknown_ids = sorted(output_ids - set(request_map))
  if unknown_ids:
    preview = ", ".join(unknown_ids[:5])
    raise batch_metrics.BatchMetricsError(
      "Could not reconstruct request metadata for batch custom_id values: "
      f"{preview}. Pass --request-map from the original phase artifacts."
    )
  return request_map


def load_or_reconstruct_request_map(
  args: argparse.Namespace,
  records: list[batch_metrics.ConversationRecord],
  output_jsonl: Path,
) -> dict[str, dict[str, Any]]:
  request_map_path = infer_request_map(args)
  if request_map_path is not None:
    return batch_metrics.load_request_map(request_map_path)
  return reconstructed_panas_before_request_map(records, output_jsonl)


def write_outputs(
  output_dir: Path,
  parsed_outputs: dict[str, dict[str, Any]],
  *,
  overwrite: bool,
  dry_run: bool,
) -> tuple[list[str], list[str]]:
  written: list[str] = []
  skipped: list[str] = []
  for rel_path, payload in sorted(parsed_outputs.items()):
    target = output_dir / rel_path
    if (
      target.is_file()
      and not overwrite
      and batch_metrics.output_is_complete("panas_before", target)
    ):
      skipped.append(rel_path)
      continue
    written.append(rel_path)
    if not dry_run:
      batch_metrics.atomic_write_json(target, payload)
  return written, skipped


def main(argv: Iterable[str] | None = None) -> int:
  args = parse_args(argv)
  try:
    if args.max_iter < 1:
      raise batch_metrics.BatchMetricsError(f"--max-iter must be positive, got {args.max_iter}")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_jsonl = infer_output_jsonl(args).resolve()
    if not output_jsonl.is_file():
      raise batch_metrics.BatchMetricsError(f"Batch output JSONL does not exist: {output_jsonl}")

    records = batch_metrics.discover_conversations(input_dir)
    records_by_filename = {record.filename: record for record in records}
    request_map = load_or_reconstruct_request_map(args, records, output_jsonl)
    parsed_outputs, parse_report = batch_metrics.parse_phase_outputs(
      output_jsonl,
      request_map,
      records_by_filename,
      args.max_iter,
      args.input_cost_per_1m,
      args.output_cost_per_1m,
    )

    batch_metrics.ensure_output_dirs(output_dir)
    written, skipped = write_outputs(
      output_dir,
      parsed_outputs,
      overwrite=args.overwrite,
      dry_run=args.dry_run,
    )

    report = {
      **parse_report,
      "written": written,
      "skipped_complete": skipped,
      "dry_run": args.dry_run,
    }
    if args.phase_dir is not None and not args.dry_run:
      batch_metrics.atomic_write_json(args.phase_dir / "recovery_parse_report.json", report)

  except batch_metrics.BatchMetricsError as exc:
    print(f"Recovery failed: {exc}", file=sys.stderr)
    return 1

  action = "Would write" if args.dry_run else "Wrote"
  print(f"{action} {len(written)} {args.phase} output file(s) to: {output_dir / args.phase}")
  if skipped:
    print(f"Skipped {len(skipped)} already-complete file(s).")
  if args.phase_dir is not None and not args.dry_run:
    print(f"Recovery report: {args.phase_dir / 'recovery_parse_report.json'}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
