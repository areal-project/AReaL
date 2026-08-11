# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.multi_gpu
@pytest.mark.skip(
    reason="requires live scheduler guards; run in the MOPD hardware gate"
)
def test_mopd_fork_lifecycle_ten_cycles():
    """Hardware gate for ten real fork/init/destroy/kill cycles."""
