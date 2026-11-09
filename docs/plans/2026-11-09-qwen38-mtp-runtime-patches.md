# Automatic Qwen3.8-27B SGLang runtime patches

AReaL's existing SGLang bootstrap reads the model configuration before executing the
selected server entry point. Matching the Qwen3.8-27B architecture automatically
installs and verifies two patches; no environment switch or separate SGLang branch is
needed. Local renamed checkpoints and Hub configurations use the same detection path.

Detection requires `model_type=qwen3_5` and the verified text configuration: hidden size
5120, intermediate size 17408, 64 layers, 24 attention heads, 4 KV heads and vocabulary
size 248320. This identifies the architecture, not a model's training provenance. MTP
layer counts and speculative decoding arguments are deliberately not part of the gate:
MTP ON and OFF comparisons should use the same runtime fixes.

| Patch                              | Behavior                                                                                                                                            |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Nonblocking speculative mRoPE copy | Removes the explicit blocking copy in the multimodal branch; the existing text-only fast path is unchanged.                                         |
| External abort completion          | Checks in-flight output for natural completion before reporting ordinary external cancellation. Internal errors and timeout aborts retain priority. |

The implementations reproduce SGLang commits `e055262d6d` and `d782649682` byte for byte
from baseline `afe21bcddb`. The existing source installer validates all four files
before writing, serializes concurrent starts, and accepts already-patched files without
writing or creating a lock. Other source revisions fail explicitly instead of silently
running without the requested patches. Supporting another fork requires validating its
source and adding its fingerprint and transformation.

The installer updates the selected interpreter's SGLang source installation; use a
job-private writable installation or a prepatched image. An unpatched read-only image
cannot be modified and produces a clear startup error. Because source files are shared
within an installation, subsequent users of the same installation also see the patches;
the model gate controls installation, not per-request behavior in a shared server.

The existing native, AReaL v2 and AWEX entry-point wrappers already pass through this
bootstrap. This change does not enable MTP, train an MTP head, change checkpoint saving,
or alter launch/scheduling policies. The HF frozen-MTP saving change remains separate.

Validation covers automatic selection, renamed checkpoints, MTP ON/OFF, other model
shapes, idempotence, concurrent starts, unknown-source rejection before writes and abort
priority. The production transforms are also checked against the full SGLang baseline
sources. These checks establish patch installation and CPU behavior; end-to-end GPU
speedup and RL reward require separate measurement.

The targeted CPU suite passes 58 tests. A separate run using the actual Qwen3.8-27B
configuration and four full upstream source files verifies automatic selection, exact
patched-file fingerprints and a write-free repeated startup. No GPU job was launched.
