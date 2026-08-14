#!/usr/bin/env python3
"""Hard import gate for BigCodeBench evaluation dependencies.

This intentionally imports every module instead of only checking package
metadata: broken wheels, ABI mismatches, and transitive import errors must stop
an expensive GPU experiment before any model server is started.
"""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
import traceback
from pathlib import Path
from importlib import metadata

MODULES = [
    "appdirs", "fire", "multipledispatch", "pqdm", "tempdir", "termcolor",
    "tree_sitter", "tree_sitter_languages", "wget",
    "bs4", "blake3", "chardet", "cryptography", "django", "dns",
    "docxtpl", "faker", "flask", "flask_login", "flask_restful",
    "flask_wtf", "flask_mail", "folium", "gensim", "geopandas", "geopy",
    "holidays", "Levenshtein", "librosa", "lxml", "matplotlib",
    "mechanize", "natsort", "networkx", "nltk", "numba", "numpy", "cv2",
    "openpyxl", "pandas", "PIL", "prettytable", "psutil", "Crypto",
    "pyfakefs", "pyquery", "pytesseract", "pytest", "python_http_client",
    "dateutil", "docx", "pytz", "yaml", "requests_mock", "requests", "rsa",
    "skimage", "sklearn", "scipy", "seaborn", "selenium", "sendgrid",
    "shapely", "soundfile", "statsmodels", "sympy", "textblob",
    "texttable", "werkzeug", "wikipedia", "wordcloud", "wordninja",
    "wtforms", "xlrd", "xlwt", "xmltodict",
    # Runtime dependencies that previously degraded silently after this gate.
    "langchain_text_splitters", "fastembed",
]


def main() -> int:
    print(f"[BCB PREFLIGHT] python={sys.executable} version={sys.version.split()[0]}")

    # Versions that are known to silently break the image's transformers,
    # SGLang, TensorFlow, or mem0 stack. Check metadata before broad imports so
    # dependency drift is reported explicitly rather than as an obscure import.
    from packaging.version import Version

    version_rules = {
        "numpy": lambda v: v == Version("1.26.4"),
        "protobuf": lambda v: Version("5.29.6") <= v < Version("7"),
        "huggingface-hub": lambda v: Version("0.34") <= v < Version("1"),
        "tokenizers": lambda v: Version("0.22") <= v <= Version("0.23.0"),
        "openai": lambda v: v == Version("2.6.1"),
        "fsspec": lambda v: v <= Version("2025.10.0"),
        "pillow": lambda v: Version("10.3") <= v < Version("12"),
        "tensorboard": lambda v: Version("2.17") <= v < Version("2.18"),
    }
    bad_versions = []
    for dist, predicate in version_rules.items():
        try:
            raw = metadata.version(dist)
            parsed = Version(raw)
            if not predicate(parsed):
                bad_versions.append((dist, raw))
                print(f"[BCB PREFLIGHT] FAIL version {dist}={raw}", file=sys.stderr)
            else:
                print(f"[BCB PREFLIGHT] OK version {dist}={raw}")
        except BaseException as exc:
            bad_versions.append((dist, f"{type(exc).__name__}: {exc}"))
            print(f"[BCB PREFLIGHT] FAIL version {dist}: {exc}", file=sys.stderr)
    if bad_versions:
        print("[FATAL] incompatible runtime package versions:", file=sys.stderr)
        for dist, detail in bad_versions:
            print(f"  - {dist}: {detail}", file=sys.stderr)
        return 1
    print(f"[BCB PREFLIGHT] PYTHONPATH={os.environ.get('PYTHONPATH', '')}")

    # A multiprocessing child using spawn re-executes the application entry
    # point as ``__mp_main__``. It must not import MemoryService/MemOS there.
    entrypoint = Path(__file__).resolve().parents[1] / "run" / "run_bcb.py"
    spawn_bootstrap = subprocess.run(
        [sys.executable, "-c", (
            "import runpy,sys; before=set(sys.modules); "
            f"runpy.run_path({str(entrypoint)!r}, run_name='__mp_main__'); "
            "loaded=sorted(n for n in set(sys.modules)-before "
            "if n=='memos' or n.startswith('memos.')); "
            "assert not loaded, loaded; print('ok')"
        )],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=os.environ.copy(), timeout=30, check=False,
    )
    if spawn_bootstrap.returncode != 0:
        print("[FATAL] run_bcb spawn-bootstrap isolation failed", file=sys.stderr)
        print(spawn_bootstrap.stdout[-2000:], file=sys.stderr)
        print(spawn_bootstrap.stderr[-4000:], file=sys.stderr)
        return 1
    print("[BCB PREFLIGHT] OK run_bcb spawn-bootstrap isolation")

    failures: list[tuple[str, str]] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
            print(f"[BCB PREFLIGHT] OK {name}")
        except BaseException as exc:  # import can raise non-Exception failures too
            detail = f"{type(exc).__name__}: {exc}"
            failures.append((name, detail))
            print(f"[BCB PREFLIGHT] FAIL {name}: {detail}", file=sys.stderr)
            traceback.print_exc(limit=4)

    if failures:
        print(
            f"[FATAL] missing/broken BCB modules ({len(failures)}/{len(MODULES)}):",
            file=sys.stderr,
        )
        for name, detail in failures:
            print(f"  - {name}: {detail}", file=sys.stderr)
        return 1

    # Import-only is insufficient for MemoryOS: the package can be present while
    # its top-level optional imports fail, causing a silent SimpleTextSplitter
    # fallback. Instantiate and execute the exact chunkers used by MemoryOS.
    try:
        from memos.chunkers.charactertext_chunker import CharacterTextChunker
        from memos.chunkers.markdown_chunker import MarkdownChunker

        char_chunker = CharacterTextChunker(chunk_size=32, chunk_overlap=4)
        md_chunker = MarkdownChunker(chunk_size=32, chunk_overlap=4, recursive=True)
        if not char_chunker.chunk("alpha beta gamma delta epsilon zeta"):
            raise RuntimeError("CharacterTextChunker returned no chunks")
        if not md_chunker.chunk("# Header\nalpha beta gamma delta epsilon zeta"):
            raise RuntimeError("MarkdownChunker returned no chunks")
        print("[BCB PREFLIGHT] OK MemoryOS langchain chunker smoke test")
    except BaseException as exc:
        print(
            f"[FATAL] MemoryOS langchain chunker smoke test failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(limit=8)
        return 1

    # fastembed can import while its Qdrant/bm25 resource is unavailable. Mem0
    # catches that initialization error and silently falls back to dense-only
    # search, which changes benchmark semantics. Require a real offline encode.
    try:
        from fastembed import SparseTextEmbedding

        encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
        result = list(encoder.embed(["strict preflight bm25 smoke test"]))
        if len(result) != 1 or len(result[0].indices) == 0:
            raise RuntimeError("Qdrant/bm25 returned an empty sparse vector")
        print(
            f"[BCB PREFLIGHT] OK Qdrant/bm25 offline encode "
            f"({len(result[0].indices)} sparse terms)"
        )
    except BaseException as exc:
        print(
            f"[FATAL] fastembed Qdrant/bm25 runtime smoke test failed: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        traceback.print_exc(limit=8)
        return 1

    # TensorFlow belongs to a separate protobuf<5 interpreter. Importing it in
    # this Mem0 process (protobuf>=5) is itself a configuration error. Capture
    # stderr and reject the exact corruption signatures that the old gate missed.
    import json
    eval_pythonpath = os.environ.get("BCB_EVAL_PYTHONPATH", "").strip()
    if not eval_pythonpath:
        print("[FATAL] BCB_EVAL_PYTHONPATH is not configured", file=sys.stderr)
        return 1
    eval_env = os.environ.copy()
    eval_env["PYTHONPATH"] = eval_pythonpath
    eval_env["PYTHONNOUSERSITE"] = "1"
    eval_env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
    ):
        eval_env[name] = "1"
    smoke = subprocess.run(
        [sys.executable, "-c", (
            "import json, tensorflow as tf; "
            "from google.protobuf import __version__ as pv; "
            "from importlib.metadata import version; "
            "assert int(pv.split('.')[0]) < 5, pv; "
            "x=tf.constant([1,2,3]); "
            "assert int(tf.reduce_sum(x).numpy()) == 6; "
            "print(json.dumps({'tensorflow':tf.__version__,"
            "'protobuf':pv,'tensorboard':version('tensorboard')}))"
        )],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=eval_env, timeout=90, check=False,
    )
    bad_markers = [
        "MessageFactory' object has no attribute 'GetPrototype",
        "cannot import name 'notf' from 'tensorboard.compat'",
        "Check failed:", "pthread_create() failed", "Thread tf_",
    ]
    marker = next((m for m in bad_markers if m in smoke.stderr), None)
    if smoke.returncode != 0 or marker:
        print("[FATAL] isolated TensorFlow evaluator smoke failed", file=sys.stderr)
        print(smoke.stdout[-2000:], file=sys.stderr)
        print(smoke.stderr[-4000:], file=sys.stderr)
        return 1
    print(f"[BCB PREFLIGHT] OK isolated TensorFlow evaluator: {smoke.stdout.strip()}")

    # Exercise the exact JSON subprocess protocol used for every scored item.
    protocol_payload = {
        "code": "def task_func():\n    return 1\n",
        "test_code": "import unittest\nclass TestCases(unittest.TestCase):\n    def test_ok(self): self.assertEqual(task_func(), 1)\n",
        "entry_point": "task_func", "max_as_limit": 30 * 1024,
        "max_data_limit": 30 * 1024, "max_stack_limit": 10,
        "min_time_limit": 1.0, "gt_time_limit": 2.0,
        "bcb_repo": str((Path(__file__).resolve().parents[1] / "3rdparty" / "bigcodebench-main")),
    }
    protocol = subprocess.run(
        [sys.executable, os.environ.get("BCB_EVAL_WORKER", "")],
        input=json.dumps(protocol_payload), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=eval_env,
        timeout=30, check=False,
    )
    try:
        protocol_result = json.loads(protocol.stdout.strip().splitlines()[-1])
    except BaseException:
        protocol_result = {}
    protocol_diagnostics = (protocol.stdout or "") + "\n" + (protocol.stderr or "")
    if (
        protocol.returncode != 0
        or not protocol_result.get("ok")
        or protocol_result.get("stat") != "pass"
        or any(marker in protocol_diagnostics for marker in bad_markers)
    ):
        print("[FATAL] isolated BCB evaluator protocol smoke failed", file=sys.stderr)
        print(protocol.stdout[-2000:], file=sys.stderr)
        print(protocol.stderr[-4000:], file=sys.stderr)
        return 1
    print("[BCB PREFLIGHT] OK isolated BCB evaluator protocol")

    # multiprocessing.Pool serializes top-level functions by module name.  The
    # evaluator must provide an importable on-disk __test__.py to spawned or
    # forkserver children; otherwise valid solutions are silently timed out.
    multiprocessing_payload = dict(protocol_payload)
    multiprocessing_payload.update({
        "code": (
            "import multiprocessing as mp\n"
            "def square(x):\n"
            "    return x * x\n"
            "def task_func(xs):\n"
            "    with mp.get_context('spawn').Pool(2) as pool:\n"
            "        return pool.map(square, xs)\n"
        ),
        "test_code": (
            "import unittest\n"
            "class TestCases(unittest.TestCase):\n"
            "    def test_pool(self):\n"
            "        self.assertEqual(task_func([1, 2, 3]), [1, 4, 9])\n"
        ),
        "min_time_limit": 2.0,
        "gt_time_limit": 10.0,
    })
    multiprocessing_smoke = subprocess.run(
        [sys.executable, os.environ.get("BCB_EVAL_WORKER", "")],
        input=json.dumps(multiprocessing_payload), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=eval_env,
        timeout=30, check=False,
    )
    try:
        multiprocessing_result = json.loads(
            multiprocessing_smoke.stdout.strip().splitlines()[-1]
        )
    except BaseException:
        multiprocessing_result = {}
    multiprocessing_diagnostics = (
        (multiprocessing_smoke.stdout or "") + "\n" +
        (multiprocessing_smoke.stderr or "")
    )
    multiprocessing_bad_markers = (
        "No module named '__test__'", "SpawnPoolWorker", "Traceback",
        "Check failed:", "pthread_create() failed", "Thread tf_",
    )
    if (
        multiprocessing_smoke.returncode != 0
        or not multiprocessing_result.get("ok")
        or multiprocessing_result.get("stat") != "pass"
        or any(marker in multiprocessing_diagnostics
               for marker in multiprocessing_bad_markers)
    ):
        print(
            "[FATAL] isolated BCB multiprocessing evaluator smoke failed",
            file=sys.stderr,
        )
        print(multiprocessing_smoke.stdout[-2000:], file=sys.stderr)
        print(multiprocessing_smoke.stderr[-4000:], file=sys.stderr)
        return 1
    print("[BCB PREFLIGHT] OK isolated BCB multiprocessing evaluator")

    # Verify the official mem0 switch was applied before import; merely setting
    # an unknown PostHog variable would not prevent remote feature-flag calls.
    from mem0.memory import telemetry as mem0_telemetry
    if mem0_telemetry.MEM0_TELEMETRY:
        print("[FATAL] MEM0 telemetry is still enabled", file=sys.stderr)
        return 1
    print("[BCB PREFLIGHT] OK mem0 telemetry disabled")

    # Import the exact TensorBoard entry point used by BCBRunner in a fresh
    # main-runtime interpreter.  It must not import real TensorFlow or emit any
    # protobuf compatibility diagnostics; an import-only module gate misses this
    # because tensorboard.compat.tf is lazy.
    main_env = os.environ.copy()
    main_env["PYTHONNOUSERSITE"] = "1"
    tb_smoke = subprocess.run(
        [sys.executable, "-c", (
            "import json, sys; "
            "from torch.utils.tensorboard import SummaryWriter; "
            "print(json.dumps({'summary_writer': SummaryWriter.__module__, "
            "'tensorflow_loaded': 'tensorflow' in sys.modules}))"
        )],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=main_env, timeout=60, check=False,
    )
    tb_bad_markers = [
        "MessageFactory", "GetPrototype",
        "cannot import name 'notf'",
        "tensorflow/core/", "Unable to register cuFFT",
        "Unable to register cuDNN", "Unable to register cuBLAS",
    ]
    tb_marker = next(
        (m for m in tb_bad_markers if m.lower() in (tb_smoke.stderr or "").lower()),
        None,
    )
    try:
        tb_result = json.loads(tb_smoke.stdout.strip().splitlines()[-1])
    except BaseException:
        tb_result = {}
    if (
        tb_smoke.returncode != 0
        or tb_marker
        or tb_result.get("tensorflow_loaded") is not False
    ):
        print("[FATAL] main-runtime TensorBoard isolation smoke failed", file=sys.stderr)
        print(tb_smoke.stdout[-2000:], file=sys.stderr)
        print(tb_smoke.stderr[-4000:], file=sys.stderr)
        return 1
    print(f"[BCB PREFLIGHT] OK main-runtime TensorBoard isolation: {tb_result}")

    print(f"[BCB PREFLIGHT] all {len(MODULES)} main-runtime modules import successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
