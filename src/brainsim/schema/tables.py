"""Canonical connectivity table (T6): one schema for every loader/builder.

Columns: (pre_id, post_id, weight, sign, cell_type, dataset, version).
weight >= 0 by convention; polarity lives in `sign` in {-1, +1} (T2).
Parquet I/O is pyarrow-gated: pyarrow is not a house dependency yet, so this
module must import and validate without it. Callers that need on-disk
round-trips get a loud RuntimeError instead of a silent no-op.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    HAVE_PYARROW = True
except ImportError:  # pragma: no cover - exercised only when pyarrow absent
    pa = None
    pq = None
    HAVE_PYARROW = False

CANONICAL_COLUMNS: tuple[str, ...] = (
    "pre_id",
    "post_id",
    "weight",
    "sign",
    "cell_type",
    "dataset",
    "version",
)


def _require_pyarrow() -> None:
    if not HAVE_PYARROW:
        raise RuntimeError(
            "pyarrow is not installed; parquet I/O is unavailable. "
            "In-memory ConnectivityTable construction/validation still works."
        )


@dataclass
class ConnectivityTable:
    """Canonical connectivity table. Every field is a 1D numpy array of equal
    length; `cell_type`/`dataset`/`version` are object arrays of str.
    """

    pre_id: np.ndarray
    post_id: np.ndarray
    weight: np.ndarray
    sign: np.ndarray
    cell_type: np.ndarray
    dataset: np.ndarray
    version: np.ndarray

    def __post_init__(self) -> None:
        self.pre_id = np.asarray(self.pre_id, dtype=np.int64)
        self.post_id = np.asarray(self.post_id, dtype=np.int64)
        self.weight = np.asarray(self.weight, dtype=np.float32)
        self.sign = np.asarray(self.sign, dtype=np.int8)
        self.cell_type = np.asarray(self.cell_type, dtype=object)
        self.dataset = np.asarray(self.dataset, dtype=object)
        self.version = np.asarray(self.version, dtype=object)

        lengths = {name: len(getattr(self, name)) for name in CANONICAL_COLUMNS}
        if len(set(lengths.values())) > 1:
            raise ValueError(f"ConnectivityTable column length mismatch: {lengths}")
        if self.n_synapses > 0:
            if np.any(self.weight < 0):
                raise ValueError("ConnectivityTable.weight must be >= 0 (T2); polarity is `sign`")
            if not np.all(np.isin(self.sign, (-1, 1))):
                raise ValueError("ConnectivityTable.sign must be in {-1, +1}")

    @property
    def n_synapses(self) -> int:
        return len(self.pre_id)

    def to_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in CANONICAL_COLUMNS}

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray]) -> "ConnectivityTable":
        missing = [c for c in CANONICAL_COLUMNS if c not in data]
        if missing:
            raise ValueError(f"ConnectivityTable.from_dict: missing columns {missing}")
        return cls(**{name: data[name] for name in CANONICAL_COLUMNS})

    def write_parquet(self, path: str | Path) -> None:
        _require_pyarrow()
        table = pa.table(self.to_dict())
        pq.write_table(table, str(path))

    @classmethod
    def read_parquet(cls, path: str | Path) -> "ConnectivityTable":
        _require_pyarrow()
        table = pq.read_table(str(path))
        present = set(table.column_names)
        missing = [c for c in CANONICAL_COLUMNS if c not in present]
        if missing:
            raise ValueError(
                f"ConnectivityTable.read_parquet({path}): missing required columns {missing}; "
                f"found {sorted(present)}"
            )
        data = {name: table.column(name).to_numpy(zero_copy_only=False) for name in CANONICAL_COLUMNS}
        return cls.from_dict(data)


def validate_columns(column_names: set[str] | list[str]) -> None:
    """Fail loudly if any canonical column is absent. Extra columns are fine."""
    present = set(column_names)
    missing = [c for c in CANONICAL_COLUMNS if c not in present]
    if missing:
        raise ValueError(f"connectivity table missing required columns: {missing}")


__all__ = [f.name for f in fields(ConnectivityTable)] + [
    "CANONICAL_COLUMNS",
    "ConnectivityTable",
    "HAVE_PYARROW",
    "validate_columns",
]
