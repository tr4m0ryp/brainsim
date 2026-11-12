"""FlyWire v783 cell-type annotations: locate, verify, and load class labels.

MB membership is taken from the systematic community annotation dump of
Schlegel et al., "Whole-brain annotation and multi-connectome cell typing of
Drosophila", Nature (2024), pinned at repository tag v2.1.0 of
github.com/flyconnectome/flywire_annotations (root IDs are for the same 783
public release as <data>/connectomes/flywire/*_783.feather). Membership is
NEVER inferred from connectivity or neuropil heuristics: a neuron is in a
class iff the annotation table says so.

Class rules (verified against the published counts on 2026-08-09):
  KC   : cell_class == "Kenyon_Cell"                      -> 5177 neurons
  MBON : cell_type or hemibrain_type startswith "MBON"    ->   96 neurons
  PN   : cell_class == "ALPN" (antennal-lobe projection)  ->  685 neurons
  DAN  : cell_class == "DAN"                              ->  331 neurons
  APL  : cell_type or hemibrain_type == "APL"             ->    2 neurons
"""
from __future__ import annotations

import csv
import hashlib
import json
import urllib.request
from pathlib import Path

from brainsim.paths import DATA, display

ANNOTATIONS_TSV = DATA / "connectomes" / "flywire" / "neuron_annotations_v2.1.0.tsv"
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
    "v2.1.0/supplemental_files/Supplemental_file1_neuron_annotations.tsv"
)
ANNOTATIONS_SOURCE = "flyconnectome/flywire_annotations@v2.1.0 (Schlegel et al. 2024, Nature)"
MANIFEST = DATA / "connectomes" / "manifest.json"

# Published FlyWire v783 counts; extraction FAILS LOUDLY outside these bands
# rather than run the battery on garbage labels (honesty rail).
SANITY_BANDS = {"KC": (4000, 6500), "MBON": (60, 200), "PN": (300, 1000),
                "DAN": (200, 700), "APL": (2, 2)}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_annotations(fetch: bool = True) -> Path:
    """Return the annotation TSV path, downloading the pinned tag if absent."""
    if ANNOTATIONS_TSV.exists():
        return ANNOTATIONS_TSV
    if not fetch:
        raise FileNotFoundError(
            f"{ANNOTATIONS_TSV} missing and fetch disabled; MB membership "
            "cannot be established without the annotation table")
    ANNOTATIONS_TSV.parent.mkdir(parents=True, exist_ok=True)
    tmp = ANNOTATIONS_TSV.with_suffix(".tsv.part")
    urllib.request.urlretrieve(ANNOTATIONS_URL, tmp)  # noqa: S310 (pinned https URL)
    tmp.rename(ANNOTATIONS_TSV)
    _record_in_manifest(ANNOTATIONS_TSV)
    return ANNOTATIONS_TSV


def _record_in_manifest(path: Path) -> None:
    """Append the annotation artifact to the connectome manifest (idempotent)."""
    if not MANIFEST.exists():
        return
    manifest = json.loads(MANIFEST.read_text())
    rel = display(path)
    if any(f.get("path") == rel for f in manifest.get("files", [])):
        return
    manifest.setdefault("files", []).append({
        "path": rel,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "source": ANNOTATIONS_URL,
        "note": "cell-type annotations pinned at tag v2.1.0 (Schlegel et al. 2024)",
    })
    MANIFEST.write_text(json.dumps(manifest, indent=1))


def load_class_map(tsv_path: Path | None = None) -> dict[str, list[int]]:
    """Parse the TSV into {class: sorted root_id list} for the five MB classes."""
    tsv_path = tsv_path or ensure_annotations()
    classes: dict[str, list[int]] = {k: [] for k in SANITY_BANDS}
    with tsv_path.open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = int(row["root_id"])
            cc = row["cell_class"]
            ct = row.get("cell_type") or ""
            hb = row.get("hemibrain_type") or ""
            if cc == "Kenyon_Cell":
                classes["KC"].append(rid)
            elif ct.startswith("MBON") or hb.startswith("MBON"):
                classes["MBON"].append(rid)
            elif cc == "ALPN":
                classes["PN"].append(rid)
            elif cc == "DAN":
                classes["DAN"].append(rid)
            elif ct == "APL" or hb == "APL":
                classes["APL"].append(rid)
    for name, ids in classes.items():
        lo, hi = SANITY_BANDS[name]
        if not lo <= len(ids) <= hi:
            raise RuntimeError(
                f"annotation sanity failed: {name} count {len(ids)} outside "
                f"published band [{lo}, {hi}] -- refusing to run on suspect labels")
        ids.sort()
    return classes


__all__ = ["ANNOTATIONS_TSV", "ANNOTATIONS_SOURCE", "ANNOTATIONS_URL",
           "SANITY_BANDS", "ensure_annotations", "load_class_map", "sha256_of"]
