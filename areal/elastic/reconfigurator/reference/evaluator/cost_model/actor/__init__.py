# SPDX-License-Identifier: Apache-2.0

"""Actor backend selection; imports are deferred until a model is queried."""


def backend_for(name):
    if name == "fsdp":
        from .fsdp import FSDPCosts

        return FSDPCosts()
    if name == "megatron":
        from .megatron import MegatronCosts

        return MegatronCosts()
    if name == "archon":
        from .archon import ArchonCosts

        return ArchonCosts()
    raise ValueError(f"Unknown Actor cost backend: {name}")
