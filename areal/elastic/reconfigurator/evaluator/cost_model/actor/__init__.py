# SPDX-License-Identifier: Apache-2.0

"""Megatron execution costs, imported on demand."""


def backend_for(name):
    if name != "megatron":
        raise ValueError("Reference Actor cost model supports Megatron")
    from .megatron import MegatronCosts

    return MegatronCosts()
