from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def normalize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    return cleaned.strip("_")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build competition ZIP from TeamID_TeamName template."
    )
    parser.add_argument("--team-id", required=True, help="Team ID from event portal")
    parser.add_argument("--team-name", required=True, help="Team name")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    template = root / "TeamID_TeamName"
    if not template.exists():
        raise FileNotFoundError(f"Template folder not found: {template}")

    team_id = normalize(args.team_id)
    team_name = normalize(args.team_name)
    if not team_id or not team_name:
        raise ValueError("Team ID and Team Name must contain at least one valid character.")

    submission_dir_name = f"{team_id}_{team_name}"
    submission_root = root / submission_dir_name

    if submission_root.resolve() == template.resolve():
        raise ValueError(
            "Use your real team id/name; output folder cannot be TeamID_TeamName."
        )

    if submission_root.exists():
        shutil.rmtree(submission_root)
    shutil.copytree(template, submission_root)

    zip_stem = f"{submission_dir_name}_GenderClassification"
    zip_path = root / f"{zip_stem}.zip"
    if zip_path.exists():
        zip_path.unlink()

    shutil.make_archive(str(root / zip_stem), "zip", root_dir=root, base_dir=submission_dir_name)

    print(f"Created folder: {submission_root}")
    print(f"Created zip: {zip_path}")


if __name__ == "__main__":
    main()
