# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_TIR_ROOT = Path(__file__).resolve().parent.parent
if str(EXAMPLES_TIR_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_TIR_ROOT))
