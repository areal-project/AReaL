# SPDX-License-Identifier: Apache-2.0

import datetime
from enum import Enum

# =============================================================================
# Distributed Training
# =============================================================================

# For large models, generation may consume more than 7200s.
# We set a large value to avoid timeout issues during generation.
DIST_GROUP_DEFAULT_TIMEOUT = datetime.timedelta(seconds=7200)


# =============================================================================
# Memory Alignment
# =============================================================================

# Default huge page size (2 MiB) for memory allocation optimization.
DEFAULT_PAGE_SIZE_BYTES = 2 * 1024 * 1024

# Default alignment for vectorized memory operations (128-bit / 16 bytes).
# This is the standard boundary across modern CPU (SSE/AVX/NEON) and GPU architectures.
# In CUDA specifically, this applies to:
#   - Vectorized load/store instructions (float4, int4, etc.)
#   - TMA (Tensor Memory Accelerator) descriptor alignment
#   - Coalesced memory access patterns
DEFAULT_VECTORIZED_ALIGNMENT_BYTES = 16


# =============================================================================
# Proximal Log-Probability Computation Enums
# =============================================================================


class ProxLogpMethod(str, Enum):
    """Method for computing proximal policy log-probabilities in decoupled PPO.

    Attributes:
        RECOMPUTE: Standard decoupled PPO - recompute via forward pass.
        LOGLINEAR: Use log-linear approximation (skip forward pass).
        METRICS: Recompute + compute approximation metrics for evaluation.
    """

    RECOMPUTE = "recompute"
    LOGLINEAR = "loglinear"
    METRICS = "metrics"

    def skips_forward_pass(self) -> bool:
        """Return True if this method skips the forward pass (optimization enabled)."""
        return self == ProxLogpMethod.LOGLINEAR


class ProxApproxMethod(str, Enum):
    """Approximation method for proximal policy log-probabilities.

    Attributes:
        LOGLINEAR: Log-linear interpolation in log-space (geometric mean in prob space).
        LINEAR: Linear interpolation in probability space (arithmetic mean).
        ROLLOUT: Use behavior policy from rollout as-is (no approximation).
    """

    LOGLINEAR = "loglinear"
    LINEAR = "linear"
    ROLLOUT = "rollout"


# =============================================================================
# Proximal Log-Probability Backward Compatibility Aliases
# (use enum classes above for new code)
# =============================================================================

# Proximal log-probability computation methods for decoupled PPO
PROX_LOGP_METHOD_RECOMPUTE = ProxLogpMethod.RECOMPUTE.value
PROX_LOGP_METHOD_LOGLINEAR = ProxLogpMethod.LOGLINEAR.value
PROX_LOGP_METHOD_METRICS = ProxLogpMethod.METRICS.value

# List of all valid prox_logp_method values for configuration
PROX_LOGP_METHODS_ALL = [m.value for m in ProxLogpMethod]

# Approximation method names used in compute_prox_logp_approximations()
PROX_APPROX_METHOD_LOGLINEAR = ProxApproxMethod.LOGLINEAR.value
PROX_APPROX_METHOD_LINEAR = ProxApproxMethod.LINEAR.value
PROX_APPROX_METHOD_ROLLOUT = ProxApproxMethod.ROLLOUT.value

# List of all approximation methods computed for metrics comparison
PROX_APPROX_METHODS_ALL = [m.value for m in ProxApproxMethod]
