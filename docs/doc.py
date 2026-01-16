#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


def find_docs_root(start_dir: str) -> str:
    current = os.path.abspath(start_dir)
    while True:
        if os.path.isfile(os.path.join(current, "_quarto.yml")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            raise FileNotFoundError("could not find docs root with _quarto.yml")
        current = parent


def run(cmd: list[str], cwd: str | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Documentation utility CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate-api", help="Generate API docs")
    subparsers.add_parser("install", help="Install website dependencies")
    subparsers.add_parser("dev", help="Run docs dev server")
    subparsers.add_parser("build", help="Render and build website")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        docs_root = find_docs_root(script_dir)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.command == "generate-api":
        run(["fumadocs-autodoc", "tatva", "-d", "_website/lib"], cwd=docs_root)
    elif args.command == "install":
        run(["pnpm", "install"], cwd=os.path.join(docs_root, "_website"))
    elif args.command == "dev":
        run(["python", "_scripts/serve.py"], cwd=docs_root)
    elif args.command == "build":
        run(["quarto", "render"], cwd=docs_root)
        run(["pnpm", "build"], cwd=os.path.join(docs_root, "_website"))
    else:
        parser.error("Unknown command")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
