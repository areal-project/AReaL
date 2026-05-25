"""Resize oversized images in mixed_rl_v2/train_v2_0524.parquet + val_v2_0524.parquet.

Per-source max_pixels policy:
  visual_probe / visual_probe_easy / visual_probe_medium / visual_probe_hard -> 2 MPx
  all other data_sources (omnispatial, reason_map*, map_trace, deep_eyes)    -> 4 MPx

Resized copies go to
``/storage/openpsi/data/lcy_image_edit/mixed_rl_v2/images_resized_v2/`` (mirror
sub-path), originals untouched.  Both parquets are rewritten in place
(originals backed up with a ``.bak`` suffix) so each affected
``images[i].path`` points at the resized copy.

Why per-source:
  Qwen3-VL patch_size=16, merge_size=2 -> vision_tokens = ceil(w/32)*ceil(h/32).
    2 MPx -> ~2 048 vision tokens   (visual_probe is high-res OCR, often used in
                                     multi-turn tool-call chains -> tighter cap
                                     keeps room for crop tools to return more imgs)
    4 MPx -> ~4 096 vision tokens   (omnispatial / reason_map -> preserve detail)
  Both stay well inside ``max_obs_length_image=8192`` and
  ``max_prompt_length=16384`` (after system+user text ~3.5K).

Usage:
    python geo_edit/scripts/resize_oversized_images_v2_0524.py [--workers 32]
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

_VISUAL_PROBE_SOURCES = frozenset({
    "visual_probe", "visual_probe_easy", "visual_probe_medium", "visual_probe_hard",
})
MAX_PIXELS_VISUAL_PROBE = 2 * 1024 * 1024
MAX_PIXELS_DEFAULT = 4 * 1024 * 1024
JPEG_QUALITY = 92
SOURCE_ROOT_PREFIX = "/storage/openpsi/data/"
RESIZED_ROOT = Path("/storage/openpsi/data/lcy_image_edit/mixed_rl_v2/images_resized_v2")


def max_pixels_for_source(data_source: str) -> int:
    if data_source in _VISUAL_PROBE_SOURCES:
        return MAX_PIXELS_VISUAL_PROBE
    return MAX_PIXELS_DEFAULT

TRAIN_PARQUET = Path("/storage/openpsi/data/lcy_image_edit/mixed_rl_v2/train_v2_0524.parquet")
VAL_PARQUET = Path("/storage/openpsi/data/lcy_image_edit/mixed_rl_v2/val_v2_0524.parquet")


def resized_path_for(src: str) -> Path:
    """Mirror under RESIZED_ROOT, stripping SOURCE_ROOT_PREFIX."""
    if src.startswith(SOURCE_ROOT_PREFIX):
        rel = src[len(SOURCE_ROOT_PREFIX):]
    else:
        rel = src.lstrip("/")
    dst = RESIZED_ROOT / rel
    return dst.with_suffix(".jpg")


def _resize_one(src: str, max_pixels: int) -> tuple[str, str, bool, str]:
    """Resize ``src`` if it exceeds ``max_pixels``.

    Returns (src, dst, was_resized, msg).
      was_resized=True => dst written or already existed
      was_resized=False & msg="ok_small" => no resize needed, kept original
      was_resized=False & msg=<error> => failed
    """
    try:
        with Image.open(src) as im:
            w, h = im.size
            pixels = w * h
            if pixels <= max_pixels:
                return src, src, False, "ok_small"
            scale = math.sqrt(max_pixels / pixels)
            new_w = max(1, math.floor(w * scale))
            new_h = max(1, math.floor(h * scale))
            dst = resized_path_for(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists() and dst.stat().st_size > 0:
                with Image.open(dst) as existing:
                    if existing.size == (new_w, new_h):
                        return src, str(dst), True, "skip_existing"
            im_rgb = im.convert("RGB")
            im_resized = im_rgb.resize((new_w, new_h), Image.LANCZOS)
            im_resized.save(str(dst), format="JPEG", quality=JPEG_QUALITY, optimize=True)
            return src, str(dst), True, f"{w}x{h}->{new_w}x{new_h}"
    except Exception as e:
        return src, src, False, f"error: {type(e).__name__}: {e}"


def gather_paths_with_threshold(parquets: Iterable[Path]) -> dict[str, int]:
    """Walk parquets and return ``{path: max_pixels}``.

    A path that appears under multiple data_sources gets the *smaller* threshold
    (so the smallest valid image works in every context that uses it).
    """
    out: dict[str, int] = {}
    for p in parquets:
        df = pq.read_table(str(p), columns=["data_source", "images"]).to_pandas()
        for ds, imgs in zip(df["data_source"], df["images"]):
            if imgs is None:
                continue
            thr = max_pixels_for_source(str(ds))
            for img in imgs:
                if isinstance(img, dict) and img.get("path"):
                    pth = img["path"]
                    out[pth] = min(out.get(pth, thr), thr)
    return out


def rewrite_parquet(src: Path, rewrite_map: dict[str, str]) -> tuple[int, int]:
    """Backup ``src`` to ``src.bak`` and rewrite its images[].path entries.

    Returns (rows, n_paths_rewritten).
    """
    backup = src.with_suffix(src.suffix + ".bak")
    if backup.exists():
        print(f"  [WARN] backup {backup.name} exists, leaving as-is")
    else:
        shutil.copy2(src, backup)
        print(f"  backed up -> {backup.name}")

    table = pq.read_table(str(src))
    df = table.to_pandas()

    n_rewritten = 0
    new_images = []
    for imgs in df["images"]:
        if imgs is None:
            new_images.append(imgs)
            continue
        out = []
        for img in imgs:
            if isinstance(img, dict):
                pth = img.get("path")
                if pth and pth in rewrite_map and rewrite_map[pth] != pth:
                    out.append({"bytes": img.get("bytes"), "path": rewrite_map[pth]})
                    n_rewritten += 1
                else:
                    out.append(img)
            else:
                out.append(img)
        new_images.append(out)
    df["images"] = new_images

    new_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
    pq.write_table(new_table, str(src))
    return len(df), n_rewritten


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32, help="parallel workers")
    ap.add_argument("--dry-run", action="store_true", help="report only, no resize / no parquet rewrite")
    args = ap.parse_args()

    print("=== Stage 1: collect path -> max_pixels map ===")
    path_thr = gather_paths_with_threshold([TRAIN_PARQUET, VAL_PARQUET])
    print(f"  unique paths across train+val: {len(path_thr)}")
    by_threshold = Counter(path_thr.values())
    for t, n in sorted(by_threshold.items()):
        print(f"    threshold={t//1024//1024}M: {n} paths")

    print("\n=== Stage 2: stat which paths exceed their per-source threshold ===")
    t0 = time.time()
    candidates: list[tuple[str, int]] = []
    by_src: Counter = Counter()
    for p, thr in path_thr.items():
        try:
            with Image.open(p) as im:
                if im.size[0] * im.size[1] > thr:
                    candidates.append((p, thr))
                    parts = p.split("/")
                    src_label = parts[4] if p.startswith("/storage/openpsi/data/") and len(parts) > 4 else "<other>"
                    by_src[f"{src_label} (>{thr//1024//1024}M)"] += 1
        except Exception as e:
            print(f"  [SKIP] {p}: {e}")
    print(f"  oversized: {len(candidates)}  (stat took {time.time()-t0:.1f}s)")
    for src, n in by_src.most_common():
        print(f"    {src}: {n}")

    if args.dry_run:
        print("\n--dry-run, exiting.")
        return

    print(f"\n=== Stage 3: resize {len(candidates)} images (workers={args.workers}) ===")
    RESIZED_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rewrite_map: dict[str, str] = {}
    counts = Counter()
    progress_every = max(1, len(candidates) // 20)
    done = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_resize_one, p, thr): p for p, thr in candidates}
        for fut in as_completed(futures):
            src, dst, was_resized, msg = fut.result()
            counts[msg.split(":")[0]] += 1
            if was_resized and dst != src:
                rewrite_map[src] = dst
            done += 1
            if done % progress_every == 0:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(candidates) - done) / rate if rate else 0
                print(f"  [{done:>5}/{len(candidates)}]  rate={rate:.1f}/s  eta={eta:.0f}s")

    print(f"\n  done in {time.time()-t0:.1f}s")
    print(f"  status counts: {dict(counts.most_common(10))}")
    print(f"  paths to rewrite in parquets: {len(rewrite_map)}")

    print("\n=== Stage 4: rewrite parquets (with .bak backups) ===")
    for p in [TRAIN_PARQUET, VAL_PARQUET]:
        n, nr = rewrite_parquet(p, rewrite_map)
        print(f"  {p.name}: rows={n}  paths_rewritten={nr}")

    print("\n=== Stage 5: post-flight verify ===")
    for p in [TRAIN_PARQUET, VAL_PARQUET]:
        df = pq.read_table(str(p), columns=["data_source", "images"]).to_pandas()
        n_over = 0
        n_total = 0
        per_src_over: Counter = Counter()
        for ds, imgs in zip(df["data_source"], df["images"]):
            if imgs is None:
                continue
            thr = max_pixels_for_source(str(ds))
            for img in imgs:
                if isinstance(img, dict) and img.get("path"):
                    n_total += 1
                    try:
                        with Image.open(img["path"]) as im:
                            if im.size[0] * im.size[1] > thr:
                                n_over += 1
                                per_src_over[ds] += 1
                    except Exception:
                        pass
        verdict = "OK" if n_over == 0 else "FAIL"
        print(f"  {p.name}: scanned {n_total} images, oversized={n_over} [{verdict}]")
        if per_src_over:
            print(f"    remaining oversized per source: {dict(per_src_over)}")

    print("\nAll done.")


if __name__ == "__main__":
    main()
