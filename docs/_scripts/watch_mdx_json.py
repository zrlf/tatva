#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

INPUT_DIR = Path("./docs/")
OUTPUT_DIR = Path("_website/.docs/docs")

COMPETING_EXTENSIONS = {".md", ".qmd", ".ipynb"}
COPYABLE_EXTENSIONS = {".mdx", ".json"}
EXCLUDED_DIRS = {
    # general exclusions
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    ".next",
}


@dataclass(frozen=True)
class FileInfo:
    mtime: float
    size: int


def should_copy(file_path: Path) -> bool:
    if file_path.suffix not in COPYABLE_EXTENSIONS:
        return False

    base = file_path.name
    if file_path.suffix == ".mdx":
        base = file_path.stem

    for ext in COMPETING_EXTENSIONS:
        if (file_path.parent / f"{base}{ext}").exists():
            return False

    return True


def iter_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        # prevent descending into "_website"
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]

        for filename in filenames:
            yield Path(dirpath) / filename


def is_excluded_path(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    return any(part in EXCLUDED_DIRS for part in relative.parts)


def run_copy(input_dir: Path, output_dir: Path) -> None:
    for file_path in iter_files(input_dir):
        if not should_copy(file_path):
            continue

        relative_path = file_path.relative_to(input_dir)
        dest_path = output_dir / relative_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest_path)
        print(f"Copied: {relative_path}")


def scan_files(input_dir: Path) -> dict[Path, FileInfo]:
    files: dict[Path, FileInfo] = {}
    for file_path in iter_files(input_dir):
        if not should_copy(file_path):
            continue
        stat = file_path.stat()
        files[file_path.relative_to(input_dir)] = FileInfo(stat.st_mtime, stat.st_size)
    return files


def copy_file(input_dir: Path, output_dir: Path, relative_path: Path) -> None:
    src_path = input_dir / relative_path
    dest_path = output_dir / relative_path
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_path)
    print(f"Copied: {relative_path}")


def delete_file(output_dir: Path, relative_path: Path) -> None:
    dest_path = output_dir / relative_path
    if dest_path.exists():
        dest_path.unlink()
        print(f"Deleted: {relative_path}")


def watch(input_dir: Path, output_dir: Path) -> None:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError as exc:
        raise SystemExit(
            "watchdog is required for watch mode. Install with: pip install watchdog"
        ) from exc

    initial = scan_files(input_dir)
    for relative_path in initial:
        copy_file(input_dir, output_dir, relative_path)

    def handle_change(src_path: Path) -> None:
        if is_excluded_path(src_path, input_dir):
            return
        if not src_path.exists():
            return
        if not should_copy(src_path):
            return
        relative_path = src_path.relative_to(input_dir)
        copy_file(input_dir, output_dir, relative_path)

    class MdxHandler(FileSystemEventHandler):
        def on_created(self, event):
            if event.is_directory:
                return
            handle_change(Path(event.src_path))

        def on_modified(self, event):
            if event.is_directory:
                return
            handle_change(Path(event.src_path))

        def on_deleted(self, event):
            if event.is_directory:
                return
            if is_excluded_path(Path(event.src_path), input_dir):
                return
            relative_path = Path(event.src_path).relative_to(input_dir)
            delete_file(output_dir, relative_path)

        def on_moved(self, event):
            if event.is_directory:
                return
            if is_excluded_path(Path(event.src_path), input_dir):
                return
            src_relative = Path(event.src_path).relative_to(input_dir)
            delete_file(output_dir, src_relative)
            handle_change(Path(event.dest_path))

    observer = Observer()
    handler = MdxHandler()
    observer.schedule(handler, str(input_dir), recursive=True)
    observer.start()

    print(f"Watching for .mdx file changes... {input_dir} {output_dir}")

    try:
        observer.join()
    except KeyboardInterrupt:
        observer.stop()
        observer.join()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch MDX/JSON files and sync to output."
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch for changes and sync in real-time.",
    )
    parser.add_argument("input_dir", nargs="?", default=INPUT_DIR)
    parser.add_argument("output_dir", nargs="?", default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not args.watch:
        run_copy(input_dir, output_dir)
        return

    watch(input_dir, output_dir)


if __name__ == "__main__":
    main()
