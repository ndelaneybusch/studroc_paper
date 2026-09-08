"""Supervised, resumable command line for the four bounded method screens."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from .common import Store

TRACKS = {"interior": 75, "likelihood": 45, "projection": 30, "m3": 30}


def fingerprint() -> dict:
    """Hash experiment and computational dependencies, excluding generated output."""
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / "scripts/methods_exploration").glob("*.py"))
    files.extend(sorted((root / "src/studroc_paper/methods").glob("*.py")))
    files.extend(
        [
            root / "scripts/c_calibration/shapes.py",
            root / "stats/methods_exploration_spec.md",
        ]
    )
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    import fiducial_core

    for path in Path(fiducial_core.__file__).parent.glob("*.so"):
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def report(*, directory: Path) -> None:
    """Render completion, eligibility and evidence links without promoting pilots."""
    lines = ["# Bounded method exploration results", ""]
    manifest = json.loads((directory / "manifest.json").read_text())
    lines.append(
        f"Profile: **{manifest['profile']}**. "
        "Pilot results are engineering checks only."
    )
    lines.append("")
    for track in TRACKS:
        status_path = directory / track / "status.json"
        summary_path = directory / track / "summary.json"
        if not status_path.exists():
            lines.append(f"- {track}: not started.")
            continue
        status = json.loads(status_path.read_text())
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        complete = status["status"] == "complete" and summary.get("complete", False)
        eligible = (
            complete
            and manifest["profile"] == "screen"
            and bool(
                summary.get("eligible")
                or summary.get("solver_eligible")
                or summary.get("candidate", {}).get("eligible")
            )
        )
        lines.append(
            f"- **{track}**: {status['status']}; {status['elapsed_seconds']:.1f}s; "
            f"complete={complete}; eligible={eligible}. "
            f"[Records]({track}/records.jsonl), [summary]({track}/summary.json), "
            f"[log]({track}/run.log)."
        )
    lines.extend(
        [
            "",
            "Full configurations and source fingerprints are in `manifest.json`.",
            "Unfinished units do not pass a gate. "
            "Finite-library likelihood widths are inner diagnostics.",
        ]
    )
    (directory / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Run tracks under hard subprocess deadlines and preserve completed units."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["pilot", "screen"], default="pilot")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--track", choices=list(TRACKS))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--minutes",
        type=float,
        help="Override the total cap; allocations retain their proportions.",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.threads < 1 or (args.minutes is not None and args.minutes <= 0):
        parser.error("threads and minutes must be positive")
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "profile": args.profile,
        "threads": args.threads,
        "sources": fingerprint(),
        "python": sys.version,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "fiducial-core"]
        },
    }
    config_path = args.out / "manifest.json"
    if config_path.exists() and json.loads(config_path.read_text()) != manifest:
        raise ValueError(
            "Output belongs to a different source/configuration; choose a new directory"
        )
    config_path.write_text(json.dumps(manifest, indent=2))
    if args.worker:
        if args.track is None:
            parser.error("worker requires track")
        store = Store(
            directory=args.out / args.track, config={**manifest, "track": args.track}
        )
        module = importlib.import_module(f"scripts.methods_exploration.{args.track}")

        def stop_worker(signum, frame) -> None:
            """Unwind track finalizers on a supervisor deadline."""
            raise TimeoutError("Track wall-clock allocation exhausted")

        signal.signal(signal.SIGTERM, stop_worker)
        module.run(store=store, pilot=args.profile == "pilot", threads=args.threads)
        return
    tracks = [args.track] if args.track else list(TRACKS)
    total = (
        args.minutes
        if args.minutes is not None
        else (10 if args.profile == "pilot" else 180)
    )
    weight = sum(TRACKS[t] for t in tracks)
    deadline = time.monotonic() + total * 60
    for track in tracks:
        directory = args.out / track
        directory.mkdir(exist_ok=True)
        status_path = directory / "status.json"
        if (
            status_path.exists()
            and json.loads(status_path.read_text())["status"] == "complete"
        ):
            continue
        timeout = max(
            0.01,
            min(total * 60 * TRACKS[track] / weight, deadline - time.monotonic()) - 5,
        )
        started = time.monotonic()
        command = [
            sys.executable,
            "-m",
            "scripts.methods_exploration.run",
            "--worker",
            "--profile",
            args.profile,
            "--out",
            str(args.out),
            "--track",
            track,
            "--threads",
            str(args.threads),
        ]
        with (directory / "run.log").open("a") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=timeout)
                state = "complete" if code == 0 else "failed"
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                state, code = "budget_exhausted", process.returncode
        status = {
            "status": state,
            "returncode": code,
            "elapsed_seconds": time.monotonic() - started,
            "budget_seconds": timeout,
        }
        status_path.write_text(json.dumps(status, indent=2))
        print(f"{track}: {state}, {status['elapsed_seconds']:.1f}s", flush=True)
        report(directory=args.out)
    if any(
        json.loads((args.out / t / "status.json").read_text())["status"] == "failed"
        for t in tracks
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
