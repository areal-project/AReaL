# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re
import socket


def get_colocate_ip_address() -> str:
    try:
        from awex.util.common import get_ip_address

        return str(get_ip_address())
    except Exception:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.connect(("8.8.8.8", 80))
                return sock.getsockname()[0]
        except Exception:
            return socket.gethostbyname(socket.gethostname())


def get_physical_cuda_device_id(local_index: int | None = None) -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    visible_devices = [x.strip() for x in visible.split(",") if x.strip()]
    if visible_devices:
        if len(visible_devices) == 1:
            return visible_devices[0]
        if local_index is None:
            try:
                import torch

                local_index = int(torch.cuda.current_device())
            except Exception:
                local_index = 0
        if 0 <= local_index < len(visible_devices):
            return visible_devices[local_index]

    if local_index is not None:
        return str(local_index)
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.current_device())
    except Exception:
        pass
    return "0"


def device_mapping_key(ip_address: str, device_id: str) -> str:
    raw = f"{ip_address}_{device_id}"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)
