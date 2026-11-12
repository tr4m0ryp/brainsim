#!/usr/bin/env bash
# torchrun-free multi-rank launcher: file:// rendezvous (torch 2.2's TCPStore
# segfaulted on our development machine; file store works everywhere incl. NCCL on one
# node). usage: launch_ranks.sh NPROC -- <python -m brainsim.bench args...>
set -euo pipefail
NP=$1; shift
[ "${1:-}" = "--" ] && shift
RDZV=$(mktemp -u "${TMPDIR:-/tmp}/brainsim-rdzv.XXXXXX")
trap 'rm -f "$RDZV"' EXIT
PIDS=()
for R in $(seq 0 $((NP - 1))); do
  RANK=$R WORLD_SIZE=$NP LOCAL_RANK=$R \
    python3 -m brainsim.bench "$@" --init "file://$RDZV" &
  PIDS+=("$!")
done
RC=0
for p in "${PIDS[@]}"; do wait "$p" || RC=$?; done
exit $RC
