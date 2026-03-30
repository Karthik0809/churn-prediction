from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    project_root: Path

    @property
    def raw_csv(self) -> Path:
        return self.project_root / "data" / "raw" / "telco_churn.csv"

    @property
    def processed_db(self) -> Path:
        return self.project_root / "data" / "processed" / "churn_features.db"

    @property
    def sql_dir(self) -> Path:
        return self.project_root / "sql"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def assets_dir(self) -> Path:
        return self.project_root / "assets"


def get_project_root() -> Path:
    # `src/` is inside project root
    return Path(__file__).resolve().parents[1]


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)
