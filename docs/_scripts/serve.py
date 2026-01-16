#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MDX/JSON watcher, Quarto preview, and pnpm dev together."
    )
    parser.add_argument(
        "--quarto-args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to `quarto preview` (default: --no-serve).",
    )
    parser.add_argument(
        "--pnpm-args",
        nargs=argparse.REMAINDER,
        help="Extra args passed to `pnpm dev`.",
    )
    return parser.parse_args()


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    for process in processes:
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    args = parse_args()
    quarto_args = args.quarto_args or [str(ROOT_DIR), "--no-serve"]
    pnpm_args = args.pnpm_args or []

    commands: list[tuple[list[str], Path]] = [
        ([sys.executable, "_scripts/watch_mdx_json.py", "--watch"], ROOT_DIR),
        (["quarto", "preview", *quarto_args], ROOT_DIR),
        (["pnpm", "dev", *pnpm_args], ROOT_DIR.joinpath("_website")),
    ]

    processes: list[subprocess.Popen] = []

    def handle_signal(_signum, _frame):
        terminate_processes(processes)
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        for command, cwd in commands:
            processes.append(subprocess.Popen(command, cwd=cwd))

        while True:
            for process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    terminate_processes(processes)
                    return exit_code
            time.sleep(0.2)
    finally:
        terminate_processes(processes)


if __name__ == "__main__":
    raise SystemExit(main())
