from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


@dataclass(frozen=True)
class StructureConfig:
    repo_root: Path
    create_gitkeep: bool


DIRECTORIES: tuple[str, ...] = (
    "data/raw",
    "data/processed",
    "notebooks",
    "src/core",
    "src/api",
    "src/ui",
    "src/scripts",
    "tests",
)


def find_repo_root(start_path: Path) -> Path:
    """
    Locate repo root by walking upwards until pyproject.toml or .git is found.
    """
    current = start_path.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return start_path.resolve()


def ensure_directories(repo_root: Path, rel_paths: Iterable[str]) -> list[Path]:
    created: list[Path] = []
    for rel in rel_paths:
        path = repo_root / rel
        path.mkdir(parents=True, exist_ok=True)
        created.append(path)
    return created


def ensure_gitkeep(dirs: Iterable[Path]) -> None:
    """
    Create .gitkeep files to allow Git to track empty directories.
    Note: if your .gitignore ignores data/, you must add exceptions for .gitkeep.
    """
    for d in dirs:
        gitkeep = d / ".gitkeep"
        gitkeep.touch(exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize PopForecast directory structure."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Repository root. If omitted, auto-detected.",
    )
    parser.add_argument(
        "--no-gitkeep",
        action="store_true",
        help="Do not create .gitkeep files.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> StructureConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(args.repo_root).resolve() if args.repo_root else find_repo_root(Path.cwd())
    return StructureConfig(repo_root=repo_root, create_gitkeep=not bool(args.no_gitkeep))


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    created_dirs = ensure_directories(config.repo_root, DIRECTORIES)

    if config.create_gitkeep:
        ensure_gitkeep(created_dirs)

    # Minimal stdout feedback (keeps MVP simple)
    print("Created/verified directories:")
    for d in created_dirs:
        print(f"- {d.relative_to(config.repo_root)}")


if __name__ == "__main__":
    main()
