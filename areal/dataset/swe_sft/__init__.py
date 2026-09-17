# SPDX-License-Identifier: Apache-2.0

"""SWE trajectory preprocessing and SFT dataset loading.

The public entry point remains :func:`get_swe_sft_dataset`. Implementation
details are split by responsibility to keep the loader maintainable.
"""

from .pipeline import get_swe_sft_dataset

__all__ = ["get_swe_sft_dataset"]
