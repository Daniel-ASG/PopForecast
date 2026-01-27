# src/scripts/download_data.py
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from kaggle.api.kaggle_api_extended import KaggleApi


DEFAULT_KAGGLE_DATASET = "luckey01/test-data-set"  # "Spotify Tracks (2021)" on Kaggle


@dataclass(frozen=True)
class DownloadConfig:
    dataset: str
    output_dir: Path
    unzip: bool
    force: bool
    quiet: bool


def configure_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s - %(message)s")


def find_repo_root(start_path: Path) -> Path:
    """
    Find the repository root by walking upwards until a pyproject.toml or .git is found.
    """
    current = start_path.resolve()
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback: best-effort (still deterministic)
    return start_path.resolve().parents[2]


def resolve_output_dir(raw_output_dir: str) -> Path:
    """
    Resolve output dir relative to repo root unless an absolute path is provided.
    """
    output_path = Path(raw_output_dir)
    if output_path.is_absolute():
        return output_path

    repo_root = find_repo_root(Path(__file__).resolve())
    return repo_root / output_path


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def kaggle_credentials_present() -> bool:
    """
    Kaggle API authenticates via either:
      - ~/.kaggle/kaggle.json
      - KAGGLE_USERNAME and KAGGLE_KEY environment variables
    """
    env_ok = bool(os.getenv("KAGGLE_USERNAME")) and bool(os.getenv("KAGGLE_KEY"))
    file_ok = (Path.home() / ".kaggle" / "kaggle.json").exists()
    return env_ok or file_ok


def validate_kaggle_json_permissions() -> Optional[str]:
    """
    Kaggle typically expects kaggle.json to be readable only by the user on Unix.
    If we can't validate (e.g., Windows), return None.
    """
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        return None

    try:
        mode = kaggle_json.stat().st_mode & 0o777
        # 0o600 is ideal; warn if group/other have any permissions
        if mode & 0o077:
            return (
                f"Permissions for {kaggle_json} look too open (mode {oct(mode)}). "
                "Consider: chmod 600 ~/.kaggle/kaggle.json"
            )
    except OSError:
        return None

    return None


def create_kaggle_api() -> KaggleApi:
    api = KaggleApi()
    api.authenticate()
    return api


def list_files_in_dir(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.rglob("*") if p.is_file()])


def dataset_already_downloaded(output_dir: Path) -> bool:
    """
    Heuristic: if there is at least one CSV file in output_dir, assume dataset exists.
    (Keeps MVP simple and avoids relying on Kaggle metadata.)
    """
    return any(p.suffix.lower() == ".csv" for p in list_files_in_dir(output_dir))


def download_dataset(config: DownloadConfig) -> None:
    if not kaggle_credentials_present():
        raise RuntimeError(
            "Kaggle credentials not found. Provide ~/.kaggle/kaggle.json "
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )

    perm_warning = validate_kaggle_json_permissions()
    if perm_warning:
        logging.warning(perm_warning)

    ensure_directory(config.output_dir)

    if dataset_already_downloaded(config.output_dir) and not config.force:
        logging.info(
            "Dataset seems already present in %s (CSV found). Use --force to re-download.",
            config.output_dir,
        )
        return

    api = create_kaggle_api()

    logging.info("Downloading Kaggle dataset: %s", config.dataset)
    logging.info("Target directory: %s", config.output_dir)

    # Downloads a .zip by default; if unzip=True Kaggle API extracts into output_dir
    api.dataset_download_files(
        dataset=config.dataset,
        path=str(config.output_dir),
        force=config.force,
        quiet=config.quiet,
        unzip=config.unzip,
    )

    files = list_files_in_dir(config.output_dir)
    if not files:
        logging.warning("Download finished, but no files were found in %s.", config.output_dir)
        return

    csv_files = [p for p in files if p.suffix.lower() == ".csv"]
    if csv_files:
        logging.info("CSV files:")
        for p in csv_files:
            logging.info(" - %s", p.relative_to(config.output_dir))
    else:
        logging.info("Files downloaded:")
        for p in files:
            logging.info(" - %s", p.relative_to(config.output_dir))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download PopForecast raw dataset from Kaggle into data/raw/."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.getenv("POPFORECAST_KAGGLE_DATASET", DEFAULT_KAGGLE_DATASET),
        help=(
            "Kaggle dataset identifier in the form 'owner/dataset-slug'. "
            f"Default: {DEFAULT_KAGGLE_DATASET}"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.getenv("POPFORECAST_RAW_DIR", "data/raw"),
        help="Destination directory (relative to repo root unless absolute). Default: data/raw",
    )
    parser.add_argument(
        "--unzip",
        action="store_true",
        help="Unzip files after download (recommended).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce Kaggle client output.",
    )
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> DownloadConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    output_dir = resolve_output_dir(args.output_dir)

    return DownloadConfig(
        dataset=args.dataset,
        output_dir=output_dir,
        unzip=bool(args.unzip),
        force=bool(args.force),
        quiet=bool(args.quiet),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    configure_logging(config.quiet)
    download_dataset(config)


if __name__ == "__main__":
    main()
