# FlashInfer source fixture

`0.6.7.post3.zip` contains unmodified `data/include/flashinfer/topk.cuh`,
`jit/sampling.py`, and `jit/topk.py` from the `flashinfer-python==0.6.7.post3` wheel
pinned in `uv.lock`, plus its Apache-2.0 LICENSE. The full wheel SHA-256 is
`9d3f1aa0313cf9e5cf99f7560b8e003c57876088c8abfe7a3d330cccd4873052`.

Source:
[FlashInfer 0.6.7.post3 on PyPI](https://pypi.org/project/flashinfer-python/0.6.7.post3/).

The compact archive makes runtime preparation tests independent of network access, a
FlashInfer installation, and CUDA. It exercises the production patch against the
released sources, including all three reset sites and both JIT wrappers. Update the
fixture and source profile together when changing the lockfile pin. The legacy profile
remains available for previously supported experiment images.
