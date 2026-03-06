"""Runtime benchmark helper for Dash startup and comparison pipeline."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _read_stream_lines(stream, queue: Queue[str]) -> None:
    for line in iter(stream.readline, ""):
        queue.put(line.rstrip("\n"))


def _wait_for_boot(
    proc: subprocess.Popen[str],
    line_queue: Queue[str],
    *,
    timeout_s: float,
    needle: str,
) -> tuple[float, list[str]]:
    start = time.perf_counter()
    seen: list[str] = []
    while time.perf_counter() - start < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError("Dash process exited before startup banner.")
        try:
            line = line_queue.get(timeout=0.2)
        except Empty:
            continue
        seen.append(line)
        if needle in line:
            return time.perf_counter(), seen
    tail = "\n".join(seen[-20:])
    raise TimeoutError(f"Timeout waiting for startup banner '{needle}'.\nRecent logs:\n{tail}")


def _wait_for_http_ready(url: str, *, timeout_s: float) -> float:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as response:  # noqa: S310
                if 200 <= int(response.status) < 500:
                    return time.perf_counter()
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"Timeout waiting for HTTP endpoint: {url}")


def _stop_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def _load_sections(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if not isinstance(data, dict):
        raise ValueError(f"Unsupported sections payload: {path}")

    sections = data.get("sections")
    if isinstance(sections, list):
        return [item for item in sections if isinstance(item, dict)]

    ranges = data.get("section_ranges")
    if isinstance(ranges, list):
        out: list[dict[str, Any]] = []
        for item in ranges:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "type": item.get("section", ""),
                    "start_page": int(item.get("start", 0) or 0),
                    "end_page": int(item.get("end", item.get("start", 0)) or 0),
                    "confidence": float(item.get("confidence", 1.0) or 1.0),
                }
            )
        return out
    raise ValueError(f"No supported sections list found in: {path}")


def _measure_startup(args: argparse.Namespace) -> dict[str, Any]:
    root = _repo_root()
    command = shlex.split(args.command)
    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONPATH": f"{root / 'src'}:{base_env.get('PYTHONPATH', '')}".rstrip(":"),
            "DASH_DEBUG": "0",
            "DASH_PORT": str(args.port),
            "DOCLING_NUM_THREADS": str(args.docling_threads),
        }
    )

    boot_times: list[float] = []
    first_screen_times: list[float] = []

    for run_idx in range(1, args.runs + 1):
        started_at = time.perf_counter()
        proc = subprocess.Popen(
            command,
            cwd=root,
            env=base_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        queue: Queue[str] = Queue()
        reader = Thread(target=_read_stream_lines, args=(proc.stdout, queue), daemon=True)
        reader.start()

        try:
            boot_ts, _ = _wait_for_boot(
                proc,
                queue,
                timeout_s=args.startup_timeout,
                needle="Dash is running on",
            )
            if args.skip_first_screen_check:
                screen_ts = boot_ts
            else:
                screen_ts = _wait_for_http_ready(
                    f"http://{args.host}:{args.port}/", timeout_s=args.screen_timeout
                )
        finally:
            _stop_process(proc)

        boot_times.append(boot_ts - started_at)
        first_screen_times.append(screen_ts - started_at)
        print(
            f"[startup] run {run_idx}/{args.runs}: "
            f"T_boot={boot_times[-1]:.2f}s, T_first_screen={first_screen_times[-1]:.2f}s"
        )

    return {
        "runs": args.runs,
        "t_boot_seconds": boot_times,
        "t_first_screen_seconds": first_screen_times,
        "median_t_boot_seconds": _median(boot_times),
        "median_t_first_screen_seconds": _median(first_screen_times),
    }


def _measure_compare(args: argparse.Namespace) -> dict[str, Any]:
    if not (args.pdf_t1 and args.pdf_t2 and args.sections_t1 and args.sections_t2 and args.bank):
        return {"enabled": False}

    root = _repo_root()
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))

    from app.comparison_runner import run_comparison_with_sections

    sections_t1 = _load_sections(args.sections_t1)
    sections_t2 = _load_sections(args.sections_t2)

    durations: list[float] = []
    summaries: list[dict[str, Any]] = []
    for run_idx in range(1, args.runs + 1):
        started_at = time.perf_counter()
        result = run_comparison_with_sections(
            pdf_path_t1=args.pdf_t1,
            pdf_path_t2=args.pdf_t2,
            bank_code=args.bank,
            sections_t1=sections_t1,
            sections_t2=sections_t2,
        )
        elapsed = time.perf_counter() - started_at
        durations.append(elapsed)
        summaries.append(result.get("summary", {}))
        print(f"[compare] run {run_idx}/{args.runs}: T_compare={elapsed:.2f}s")

    return {
        "enabled": True,
        "runs": args.runs,
        "t_compare_seconds": durations,
        "median_t_compare_seconds": _median(durations),
        "summary_last_run": summaries[-1] if summaries else {},
    }


def _compare_with_baseline(current: dict[str, Any], baseline_path: str | None) -> dict[str, Any] | None:
    if not baseline_path:
        return None
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    metrics = [
        ("median_t_boot_seconds", "startup"),
        ("median_t_first_screen_seconds", "startup"),
        ("median_t_compare_seconds", "compare"),
    ]
    for metric, section in metrics:
        current_section = current.get(section, {})
        baseline_section = baseline.get(section, {})
        current_value = current_section.get(metric)
        baseline_value = baseline_section.get(metric)
        if not isinstance(current_value, (int, float)) or not isinstance(baseline_value, (int, float)):
            continue
        if baseline_value <= 0:
            continue
        delta = current_value - baseline_value
        pct = (delta / baseline_value) * 100.0
        out[metric] = {
            "baseline": baseline_value,
            "current": current_value,
            "delta_seconds": delta,
            "delta_percent": pct,
        }
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark startup and compare pipeline timings.")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for median timings.")
    parser.add_argument(
        "--command",
        default="uv run python -m app.app",
        help="Startup command used for Dash boot measurements.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Dash host for readiness checks.")
    parser.add_argument("--port", type=int, default=8050, help="Dash port for readiness checks.")
    parser.add_argument(
        "--startup-timeout", type=float, default=120.0, help="Timeout in seconds for startup banner."
    )
    parser.add_argument(
        "--screen-timeout", type=float, default=30.0, help="Timeout in seconds for first screen readiness."
    )
    parser.add_argument(
        "--skip-first-screen-check",
        action="store_true",
        help="Skip HTTP readiness check and use T_boot as T_first_screen.",
    )
    parser.add_argument(
        "--docling-threads",
        type=int,
        default=4,
        help="DOCLING_NUM_THREADS for startup command.",
    )
    parser.add_argument("--pdf-t1", help="Path to T1 PDF for compare timing.")
    parser.add_argument("--pdf-t2", help="Path to T2 PDF for compare timing.")
    parser.add_argument("--bank", help="Bank code for compare timing.")
    parser.add_argument("--sections-t1", help="JSON file with sections for T1.")
    parser.add_argument("--sections-t2", help="JSON file with sections for T2.")
    parser.add_argument("--output-json", help="Write full benchmark result to JSON file.")
    parser.add_argument("--baseline-json", help="Optional baseline JSON to print deltas.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    startup = _measure_startup(args)
    compare = _measure_compare(args)
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "startup": startup,
        "compare": compare,
    }
    deltas = _compare_with_baseline(result, args.baseline_json)
    if deltas is not None:
        result["delta_vs_baseline"] = deltas

    print("\n=== Median timings ===")
    print(f"T_boot: {startup['median_t_boot_seconds']:.2f}s")
    print(f"T_first_screen: {startup['median_t_first_screen_seconds']:.2f}s")
    if compare.get("enabled"):
        print(f"T_compare: {compare['median_t_compare_seconds']:.2f}s")
    if deltas:
        print("\n=== Delta vs baseline ===")
        for metric, payload in deltas.items():
            print(
                f"{metric}: {payload['current']:.2f}s "
                f"(baseline {payload['baseline']:.2f}s, "
                f"delta {payload['delta_seconds']:+.2f}s / {payload['delta_percent']:+.1f}%)"
            )

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nBenchmark JSON written to: {output_path}")


if __name__ == "__main__":
    main()
