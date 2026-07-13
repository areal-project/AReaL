# SPDX-License-Identifier: Apache-2.0

# Adapted from torchtitan: torchtitan/distributed/parallel_dims.py

import functools
from dataclasses import dataclass, field

import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from areal.utils import logging


@functools.cache
def _get_logger() -> logging.Logger:
    """Get rank-aware logger for this module."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    return logging.getLogger(f"[Archon ParallelDims Rank {rank}]")


__all__ = [
    "ArchonParallelDims",
]


@dataclass
class ArchonParallelDims:
    """Parallel dimensions for Archon engine.

    Archon Engine uses PyTorch-native distributed APIs inspired by torchtitan.

    Note: dp_replicate is always 1 - no HSDP support yet.

    Attributes:
        dp_shard: FSDP shard dimension (data parallel). -1 means auto-compute.
        tp: Tensor Parallel size.
        cp: Context Parallel size (implemented as Ulysses SP, not Ring Attention).
        pp: Pipeline Parallel size.
        ep: Expert Parallel size.
        etp: Expert Tensor Parallel size (must be 1 or equal to tp).
        world_size: Total number of processes.
        device_type: Device type for mesh creation (e.g., "cuda", "npu").

    Mesh semantics:
        - total_gpu = pp × dp_shard × cp × tp
        - fsdp_size = dp_shard × cp  (CP ranks participate in weight sharding)

    Expert Parallelism Strategy Selection:
        The strategy for MoE expert layers depends on EP and ETP settings:

        | EP  | TP  | etp | Strategy              | Expert Weight Sharding       |
        |-----|-----|-----|-----------------------|------------------------------|
        | 1   | 1   | -   | None                  | Replicate                    |
        | 1   | >1  | -   | TensorParallel        | [Shard(1/2)]                 |
        | >1  | 1   | -   | ExpertParallel        | [Shard(0)]                   |
        | >1  | >1  | 1   | ExpertParallel        | [Shard(0)] (TP borrowed by EP) |
        | >1  | >1  | tp  | ExpertTensorParallel  | [Shard(0), Shard(1/2)]       |

        When EP>1 and TP>1:
        - etp=1: TP dimension is borrowed by EP for token dispatch. Experts use
          only EP sharding [Shard(0)]. EP borrows from dp_shard × cp × tp.
        - etp=tp: TP dimension remains independent. Experts use 2D sharding
          [Shard(0), Shard(1/2)] combining EP and TP. EP borrows from
          dp_shard × cp only.

    Available mesh dimensions (when EP disabled):
        - pp: Pipeline Parallel
        - dp_shard: FSDP shard dimension
        - cp: Context Parallel
        - tp: Tensor Parallel
        - dp: dp_shard (for data loading)
        - dp_shard_cp: dp_shard * cp (for FSDP sharding)
        - dp_cp: dp_shard * cp (for loss all-reduce)
        - pp_cp_tp: pp * cp * tp (for context and model parallel group / data broadcast)

    Available mesh dimensions (when EP enabled):
        - pp: Pipeline Parallel
        - dp_shard_mod_ep: dp_shard * cp / ep when etp=tp, else dp_shard * cp * tp / ep
        - dp_shard_in_ep: ep / cp when etp=tp, else ep / (cp * tp)
        - cp: Context Parallel
        - tp: Tensor Parallel
        - dp: dp_shard (for data loading)
        - dp_shard_cp: dp_shard * cp (for FSDP sharding of dense params)
        - dp_cp: dp_shard * cp (for loss all-reduce)
        - pp_cp_tp: pp * cp * tp (for context and model parallel group / data broadcast)
        - ep: Expert Parallel (flattened from dp_shard_in_ep * cp * etp)
        - ep_tp: 2D mesh [ep, tp] for ExpertTensorParallel (only when etp=tp)

    Example (pp=1, dp_shard=2, cp=2, tp=2, ep=1, etp=1, 8 GPUs):
        - fsdp_size = dp_shard × cp = 2 × 2 = 4
        - Mesh dims: (pp=1, dp_shard=2, cp=2, tp=2)

    Example (pp=2, dp_shard=2, cp=1, tp=2, ep=1, etp=1, 8 GPUs):
        - fsdp_size = dp_shard × cp = 2 × 1 = 2
        - Mesh dims: (pp=2, dp_shard=2, cp=1, tp=2)

    Example (pp=1, dp_shard=2, cp=1, tp=2, ep=2, etp=1, 4 GPUs):
        - ep borrows from dp_shard_in_ep * cp * tp (since etp=1)
        - dp_shard_mod_ep = dp_shard * cp * tp / ep = 2 * 1 * 2 / 2 = 2
        - dp_shard_in_ep = ep / (cp * tp) = 2 / (1 * 2) = 1
        - Mesh dims: (pp=1, dp_shard_mod_ep=2, dp_shard_in_ep=1, cp=1, tp=2)

    Example (pp=1, dp_shard=2, cp=1, tp=2, ep=2, etp=2, 4 GPUs):
        - ep borrows from dp_shard_in_ep * cp only (since etp=tp)
        - dp_shard_mod_ep = dp_shard * cp / ep = 2 * 1 / 2 = 1
        - dp_shard_in_ep = ep / cp = 2 / 1 = 2
        - Mesh dims: (pp=1, dp_shard_mod_ep=1, dp_shard_in_ep=2, cp=1, tp=2)
        - ep_tp mesh: 2D mesh [ep=2, tp=2] for 2D expert weight sharding
    """

    dp_replicate: int = 1  # HSDP replicate dimension (not supported yet)
    dp_shard: int = -1  # FSDP shard dimension, -1 means auto
    cp: int = 1  # Context Parallel size (Ulysses SP)
    tp: int = 1  # Tensor Parallel size
    pp: int = 1  # Pipeline Parallel size
    ep: int = 1  # Expert Parallel size
    etp: int = 1  # Expert Tensor Parallel size (1 or tp)
    world_size: int = 1
    device_type: str = "cuda"

    # Internal state
    _world_mesh: DeviceMesh | None = field(default=None, repr=False)
    _meshes: dict[str, DeviceMesh] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if self.dp_shard < 0:
            self.dp_shard = self.world_size // (self.tp * self.cp * self.pp)

        expected_world_size = self.dp_shard * self.tp * self.cp * self.pp
        if expected_world_size != self.world_size:
            raise ValueError(
                f"dp_shard * tp * cp * pp must equal world_size, "
                f"got {self.dp_shard} * {self.tp} * {self.cp} * {self.pp} = {expected_world_size}, "
                f"world_size={self.world_size}"
            )

        # Validate ETP constraints
        if self.etp not in (1, self.tp):
            raise ValueError(
                f"etp must be 1 or equal to tp, got etp={self.etp}, tp={self.tp}"
            )

        # Validate EP constraints based on ETP mode
        if self.ep > 1:
            if self.etp == self.tp:
                # ETP=TP mode: EP borrows from dp_shard × cp only (not tp)
                if self.ep % self.cp != 0:
                    raise ValueError(
                        f"When etp=tp, ep must be divisible by cp, "
                        f"got ep={self.ep}, cp={self.cp}"
                    )
                if (self.dp_shard * self.cp) % self.ep != 0:
                    raise ValueError(
                        f"When etp=tp, dp_shard * cp must be divisible by ep, "
                        f"got {self.dp_shard} * {self.cp} = {self.dp_shard * self.cp}, ep={self.ep}"
                    )
            else:
                # ETP=1 mode: EP borrows from dp_shard × cp × tp
                if self.ep % (self.cp * self.tp) != 0:
                    raise ValueError(
                        f"When etp=1, ep must be divisible by cp * tp, "
                        f"got ep={self.ep}, cp={self.cp}, tp={self.tp}"
                    )
                if (self.dp_shard * self.cp * self.tp) % self.ep != 0:
                    raise ValueError(
                        f"When etp=1, dp_shard * cp * tp must be divisible by ep, "
                        f"got {self.dp_shard} * {self.cp} * {self.tp} = {self.dp_shard * self.cp * self.tp}, ep={self.ep}"
                    )

    # =========================================================================
    # Mesh Creation
    # =========================================================================

    def build_mesh(self) -> DeviceMesh:
        """Build device mesh for all parallelism dimensions."""
        if self.ep > 1:
            return self._build_mesh_with_ep()
        else:
            return self._build_mesh_without_ep()

    def _build_mesh_without_ep(self) -> DeviceMesh:
        """Build mesh when EP is disabled."""
        # Always include all dimensions, even if size=1
        # This ensures submeshes like mp (cp × tp) can always be created
        dims = [self.pp, self.dp_shard, self.cp, self.tp]
        names = ["pp", "dp_shard", "cp", "tp"]

        _get_logger().info(f"Building 4-D device mesh with {names}, {dims}")
        mesh = init_device_mesh(
            self.device_type, tuple(dims), mesh_dim_names=tuple(names)
        )

        self._meshes["pp"] = mesh["pp"]
        self._meshes["dp_shard"] = mesh["dp_shard"]
        self._meshes["cp"] = mesh["cp"]
        self._meshes["tp"] = mesh["tp"]

        # dp mesh: for data loading
        self._meshes["dp"] = mesh["dp_shard"]._flatten(mesh_dim_name="dp")

        # dp_shard_cp mesh: for FSDP param sharding
        self._meshes["dp_shard_cp"] = mesh["dp_shard", "cp"]._flatten(
            mesh_dim_name="dp_shard_cp"
        )

        # dp_cp mesh: for loss all-reduce
        self._meshes["dp_cp"] = mesh["dp_shard", "cp"]._flatten(mesh_dim_name="dp_cp")

        # pp_cp_tp mesh: for PP × CP × TP (context and model parallel group)
        self._meshes["pp_cp_tp"] = mesh["pp", "cp", "tp"]._flatten(
            mesh_dim_name="pp_cp_tp"
        )

        self._world_mesh = mesh
        return mesh

    def _build_mesh_with_ep(self) -> DeviceMesh:
        """Build mesh when EP is enabled.

        Handles both etp=1 and etp=tp cases:
        - etp=1: EP borrows from dp_shard_in_ep * cp * tp
        - etp=tp: EP borrows from dp_shard_in_ep * cp only (tp independent)
        """
        # Calculate dimensions based on ETP mode
        if self.etp == self.tp:
            # ETP=TP: ep = dp_shard_in_ep * cp (NOT including tp)
            dp_shard_mod_ep = self.dp_shard * self.cp // self.ep
            dp_shard_in_ep = self.ep // self.cp
        else:
            # ETP=1: ep = dp_shard_in_ep * cp * tp
            dp_shard_mod_ep = self.dp_shard * self.cp * self.tp // self.ep
            dp_shard_in_ep = self.ep // (self.cp * self.tp)

        dims = [self.pp, dp_shard_mod_ep, dp_shard_in_ep, self.cp, self.tp]
        names = ["pp", "dp_shard_mod_ep", "dp_shard_in_ep", "cp", "tp"]

        _get_logger().info(
            f"Building 5-D device mesh (etp={self.etp}) with {names}, {dims}"
        )
        mesh = init_device_mesh(
            self.device_type, tuple(dims), mesh_dim_names=tuple(names)
        )

        # Store base meshes
        self._meshes["pp"] = mesh["pp"]
        self._meshes["dp_shard_mod_ep"] = mesh["dp_shard_mod_ep"]
        self._meshes["dp_shard_in_ep"] = mesh["dp_shard_in_ep"]
        self._meshes["cp"] = mesh["cp"]
        self._meshes["tp"] = mesh["tp"]

        # dp mesh: for data loading
        self._meshes["dp"] = mesh["dp_shard_mod_ep", "dp_shard_in_ep"]._flatten(
            mesh_dim_name="dp"
        )

        # dp_shard_cp mesh: for FSDP param sharding
        self._meshes["dp_shard_cp"] = mesh[
            "dp_shard_mod_ep", "dp_shard_in_ep", "cp"
        ]._flatten(mesh_dim_name="dp_shard_cp")

        # dp_cp mesh: for loss all-reduce
        self._meshes["dp_cp"] = mesh[
            "dp_shard_mod_ep", "dp_shard_in_ep", "cp"
        ]._flatten(mesh_dim_name="dp_cp")

        # pp_cp_tp mesh: for PP × CP × TP (context and model parallel group)
        self._meshes["pp_cp_tp"] = mesh["pp", "cp", "tp"]._flatten(
            mesh_dim_name="pp_cp_tp"
        )

        # ep mesh: flatten based on ETP mode
        if self.etp == self.tp:
            # ETP=TP: ep = dp_shard_in_ep * cp (NOT including tp)
            # First flatten to create "ep" dimension, then create 2D ep_tp mesh
            ep_mesh_dims = ["dp_shard_in_ep"]
            if self.cp > 1:
                ep_mesh_dims.append("cp")

            if len(ep_mesh_dims) > 1:
                mesh[tuple(ep_mesh_dims)]._flatten(mesh_dim_name="ep")
            else:
                # If only dp_shard_in_ep, just rename it
                mesh["dp_shard_in_ep"]._flatten(mesh_dim_name="ep")
            self._meshes["ep"] = mesh["ep"]

            # ep_tp mesh: 2D mesh [ep, tp] for ExpertTensorParallel
            self._meshes["ep_tp"] = mesh["ep", "tp"]
        else:
            # ETP=1: ep = dp_shard_in_ep * cp * tp
            self._meshes["ep"] = mesh["dp_shard_in_ep", "cp", "tp"]._flatten(
                mesh_dim_name="ep"
            )

        self._world_mesh = mesh
        return mesh

    @property
    def world_mesh(self) -> DeviceMesh:
        """Lazily build and return the device mesh."""
        if self._world_mesh is None:
            self._world_mesh = self.build_mesh()
        return self._world_mesh

    # =========================================================================
    # Enabled Flags
    # =========================================================================

    @property
    def dp_replicate_enabled(self) -> bool:
        """Whether HSDP replication is enabled (dp_replicate > 1)."""
        return self.dp_replicate > 1

    @property
    def dp_shard_enabled(self) -> bool:
        """Whether FSDP sharding is enabled (dp_shard > 1)."""
        return self.dp_shard > 1

    @property
    def dp_enabled(self) -> bool:
        """Whether any data parallelism is enabled (dp_replicate > 1 or dp_shard > 1)."""
        return self.dp_replicate_enabled or self.dp_shard_enabled

    @property
    def dp_cp_enabled(self) -> bool:
        """Whether data or context parallelism is enabled (for loss all-reduce)."""
        return self.dp_enabled or self.cp_enabled

    @property
    def fsdp_enabled(self) -> bool:
        """Whether FSDP is enabled (dp_shard > 1 or cp > 1)."""
        return self.dp_shard_enabled or self.cp_enabled

    @property
    def cp_enabled(self) -> bool:
        """Whether context parallelism is enabled."""
        return self.cp > 1

    @property
    def tp_enabled(self) -> bool:
        """Whether tensor parallelism is enabled."""
        return self.tp > 1

    @property
    def pp_enabled(self) -> bool:
        """Whether pipeline parallelism is enabled."""
        return self.pp > 1

    @property
    def ep_enabled(self) -> bool:
        """Whether expert parallelism is enabled."""
        return self.ep > 1

    @property
    def etp_enabled(self) -> bool:
        """Whether expert tensor parallelism is enabled (etp > 1)."""
        return self.etp > 1

    # =========================================================================
    # Utilities
    # =========================================================================

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        """Gradient divide factor for FSDP (consistent with data parallelism degree).

        This is needed for FSDP-sharded experts when Expert Parallel is enabled.
        Although the FSDP sharding of experts is done on a mesh of a different size than
        other parameters, the gradient division factor should be consistent with data.
        """
        return self.dp_shard * self.cp

    @property
    def seq_len_divisor(self) -> int:
        """Minimum divisor for sequence length (for TP and CP compatibility)."""
        return self.tp * self.cp * 2

    @property
    def context_and_model_parallel_size(self) -> int:
        """Context and model parallel size (cp * tp * pp)."""
        return self.cp * self.tp * self.pp

    def get_mesh(self, name: str) -> DeviceMesh | None:
        """Get submesh by name, return None if not available.

        This method safely retrieves a submesh without throwing an exception
        if the dimension doesn't exist (e.g., cp=1 means no "cp" mesh).

        Args:
            name: Mesh dimension name. Available names depend on EP status:
                - Without EP: 'pp', 'dp', 'dp_shard_cp', 'dp_cp', 'dp_shard', 'cp', 'tp', 'pp_cp_tp'
                - With EP: 'pp', 'dp', 'dp_shard_cp', 'dp_cp', 'ep', 'dp_shard_mod_ep',
                          'dp_shard_in_ep', 'cp', 'tp', 'pp_cp_tp'

        Returns:
            DeviceMesh for the requested dimension, or None if not available.
        """
        # Ensure mesh is built
        _ = self.world_mesh
        return self._meshes.get(name)

    def get_group(self, name: str) -> dist.ProcessGroup | None:
        """Get process group by name, return None if not available.

        This method always returns a group if the mesh exists, even if size=1.
        Use explicit checks like `if self.tp_enabled` before calling this method
        if you want to skip operations when parallelism is not needed.

        Args:
            name: Mesh dimension name. Available names depend on EP status:
                - Without EP: 'pp', 'dp', 'dp_shard_cp', 'dp_cp', 'dp_shard', 'cp', 'tp', 'pp_cp_tp'
                - With EP: 'pp', 'dp', 'dp_shard_cp', 'dp_cp', 'ep', 'dp_shard_mod_ep',
                          'dp_shard_in_ep', 'cp', 'tp', 'pp_cp_tp'

        Returns:
            ProcessGroup for the requested dimension, or None if mesh not available.
        """
        submesh = self.get_mesh(name)
        if submesh is None:
            return None
        return submesh.get_group()
