#!/usr/bin/env python3
"""Select the furthest valid HLE checkpoint across one or more experiment globs."""
import argparse, glob, json, os
from pathlib import Path

ap=argparse.ArgumentParser()
ap.add_argument('--glob', action='append', required=True, dest='globs')
ap.add_argument('--max-no-id', type=int, default=0)
ap.add_argument('--require-region-manager', action='store_true')
ap.add_argument('--json', action='store_true')
a=ap.parse_args()
rows=[]
for pattern in a.globs:
    for raw in glob.glob(pattern):
        p=Path(raw)
        meta=p/'snapshot_meta.json'; state=p/'local_cache/cum_state.json'
        if not (p.is_dir() and meta.is_file() and state.is_file()): continue
        if a.require_region_manager and not (p/'local_cache/region_manager.json').is_file(): continue
        try: x=json.loads(state.read_text())
        except Exception: continue
        recs=x.get('batch_all_recs',[]) or []
        ids={str(r.get('id')) for r in recs if isinstance(r,dict) and r.get('id') is not None}
        no_id_questions={str(r.get('question','')) for r in recs if isinstance(r,dict) and r.get('id') is None}
        if len(no_id_questions)>a.max_no_id: continue
        unique=len(ids)+len(no_id_questions)
        section=int(x.get('next_section',0) or 0); batch=int(x.get('next_batch',0) or 0)
        pending=len(set(map(str,x.get('pending_task_ids',[]) or [])))
        # Higher section/progress/cursor/mtime wins; fewer pending breaks exact ties.
        score=(section,unique,batch,-pending,p.stat().st_mtime)
        rows.append({'path':str(p),'score':score,'section':section,'batch':batch,'unique':unique,'pending':pending,'no_id':len(no_id_questions),'mtime':p.stat().st_mtime})
if not rows: raise SystemExit('no valid checkpoint found')
best=max(rows,key=lambda r:r['score'])
print(json.dumps(best,ensure_ascii=False) if a.json else best['path'])
