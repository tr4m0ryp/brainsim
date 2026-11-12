#!/usr/bin/env bash
# Fetch Shiu et al.'s Drosophila whole-brain LIF model (MIT licence) at the
# commit both fly gates ran against. Its Connectivity_783.parquet and
# Completeness_783.csv are FlyWire v783 derivatives distributed by that
# repository; they are used locally and never copied into this one.
#   bash scripts/data/fetch_shiu_model.sh   # -> <data>/external/Drosophila_brain_model
set -euo pipefail
COMMIT=91bdd1e7dcf193f3e7ca5a8933497fcef63b7960
DATA_ROOT=${BRAINSIM_DATA:-"$(cd "$(dirname "$0")/../.." && pwd)/data"}
DEST="$DATA_ROOT/external/Drosophila_brain_model"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$(dirname "$DEST")"
  git clone https://github.com/philshiu/Drosophila_brain_model "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
echo "Drosophila_brain_model at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
