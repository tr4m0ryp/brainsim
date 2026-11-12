"""Torch ports of the published rules (T4) for the torch engine (T5): same
update semantics and the SAME parameter dataclasses as brainsim.rules.*,
operating on TorchCSRProjection weight tensors in place via batched gathers
(no python loops over synapses). Works identically on TorchNetwork
projections and ShardedNetwork local slices, whose shapes already carry the
global-pre/local-post convention the rules need.

torch is imported only inside this package and engine.torch_core;
brainsim.rules itself stays torch-free.
"""
from brainsim.rules.torch_rules import allocation, btsp, cascade, csr, mb_rule, replay, stdp
from brainsim.rules.torch_rules.allocation import TorchAllocationState
from brainsim.rules.torch_rules.btsp import TorchBTSPState
from brainsim.rules.torch_rules.csr import synapse_post_ids, synapse_pre_ids
from brainsim.rules.torch_rules.mb_rule import TorchMBRuleState
from brainsim.rules.torch_rules.replay import TorchReplayScheduler
from brainsim.rules.torch_rules.stdp import TorchSTDPState

__all__ = [
    "allocation",
    "btsp",
    "cascade",
    "csr",
    "mb_rule",
    "replay",
    "stdp",
    "TorchAllocationState",
    "TorchBTSPState",
    "TorchMBRuleState",
    "TorchReplayScheduler",
    "TorchSTDPState",
    "synapse_post_ids",
    "synapse_pre_ids",
]
