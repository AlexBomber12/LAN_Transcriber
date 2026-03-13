from __future__ import annotations

import argparse
from typing import Sequence

from lan_app.snippet_repair import (
    SnippetRepairError,
    backfill_missing_snippets,
    repair_recording_snippets,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate missing or stale speaker snippet artifacts.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--recording-id",
        help="Repair snippet artifacts for exactly one recording.",
    )
    target.add_argument(
        "--scan-missing",
        action="store_true",
        help="Backfill terminal recordings that are missing snippets_manifest.json.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.recording_id:
        try:
            result = repair_recording_snippets(
                args.recording_id,
                origin="cli_single",
            )
        except SnippetRepairError as exc:
            print(f"FAILED {args.recording_id} {exc.code}: {exc}")
            return 1
        print(
            "REGENERATED "
            f"{result.recording_id} status={result.manifest_status} "
            f"accepted={result.accepted_snippets} speakers={result.speaker_count} "
            f"warnings={result.warning_count} audio_source={result.audio_source}"
        )
        return 0

    summary = backfill_missing_snippets(origin="cli_batch")
    for item in summary.items:
        line = f"{item.outcome.upper()} {item.recording_id} {item.detail}"
        if item.manifest_status:
            line += f" status={item.manifest_status}"
        if item.accepted_snippets:
            line += f" accepted={item.accepted_snippets}"
        print(line)
    print(
        "SUMMARY "
        f"regenerated={summary.regenerated} "
        f"skipped={summary.skipped} "
        f"failed={summary.failed}"
    )
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
