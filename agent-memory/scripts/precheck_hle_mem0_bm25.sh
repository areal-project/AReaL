#!/bin/bash
set -euo pipefail
ROOT=/storage/openpsi/users/yl/agent-memory
LOG="$ROOT/MemRL/logs/hle_mem0_bm25_precheck_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
SP=/tmp/hle_mem0_bm25_precheck_site
DB=/tmp/hle_mem0_bm25_precheck_qdrant
SRC=/storage/openpsi/experiments/checkpoints/admin/yl-mem-region/hle/exp_hle_mem0_gemini35flash_yl-hle-mem0-g35f-20260719-185247/snapshot/1_b9/mem0_qdrant
rm -rf "$SP" "$DB"
mkdir -p "$SP"
uv pip install --python python3 --target "$SP" -i https://pypi.antfin-inc.com/simple/ 'qdrant-client[fastembed]>=1.17,<1.18'
cp -a "$SRC" "$DB"
export PYTHONPATH="$SP:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FASTEMBED_CACHE_PATH="$ROOT/MemRL/scripts/fastembed_cache"
python3 - <<'PY'
import os
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector
cache=os.environ['FASTEMBED_CACHE_PATH']
enc=SparseTextEmbedding('Qdrant/bm25', cache_dir=cache, local_files_only=True)
probe=list(enc.embed(['mem0 bm25 remote python312 verification']))[0]
assert len(probe.indices)>0
path='/tmp/hle_mem0_bm25_precheck_qdrant'
name='hle_mem0_yl_hle_mem0_g35f_20260719_185247'
c=QdrantClient(path=path)
info=c.get_collection(name)
assert info.config.params.sparse_vectors and 'bm25' in info.config.params.sparse_vectors
pts,_=c.scroll(name,limit=3,with_payload=True,with_vectors=True)
assert pts
updates=[]
query_text=None
for p in pts:
    text=(p.payload or {}).get('text_lemmatized') or (p.payload or {}).get('data','')
    if query_text is None:
        query_text=text
    sparse=list(enc.embed([text]))[0]
    dense=p.vector if isinstance(p.vector,dict) else {'':p.vector}
    dense=dict(dense); dense['bm25']=SparseVector(indices=sparse.indices.tolist(),values=sparse.values.tolist())
    updates.append(PointStruct(id=p.id,vector=dense,payload=p.payload))
c.upsert(collection_name=name,points=updates)
query_sparse=list(enc.query_embed([query_text]))[0]
hits=c.query_points(collection_name=name,query=SparseVector(indices=query_sparse.indices.tolist(),values=query_sparse.values.tolist()),using='bm25',limit=3)
assert hits.points, 'BM25 query returned no hits after sparse-vector backfill'
print(f'BM25_REMOTE_PRECHECK_OK python_points={len(pts)} updated={len(updates)} hits={len(hits.points)} cache={cache}')
c.close()
PY
echo "PRECHECK_LOG=$LOG"
