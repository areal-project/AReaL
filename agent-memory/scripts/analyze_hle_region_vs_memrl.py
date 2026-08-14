#!/usr/bin/env python3
"""Paired HLE Region-vs-MemRL audit from partial checkpoint states."""
import argparse, json, collections
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--region-state', required=True)
ap.add_argument('--memrl-state', required=True)
ap.add_argument('--dataset', default='data/hle/hle_test.parquet')
a=ap.parse_args()

def load(path):
    d=json.loads(Path(path).read_text())
    return {str(x['id']): bool(x['correct']) for x in d.get('batch_all_recs',[]) if x.get('id') is not None}
r,m=load(a.region_state),load(a.memrl_state)
df=pd.read_parquet(a.dataset)
meta={str(x['id']):x for x in df.to_dict('records')}
ids=set(r)&set(m)
groups={
 'memrl_only':[i for i in ids if m[i] and not r[i]],
 'region_only':[i for i in ids if r[i] and not m[i]],
 'both_right':[i for i in ids if r[i] and m[i]],
 'both_wrong':[i for i in ids if not r[i] and not m[i]],
}
print('intersection',len(ids),'region_correct',sum(r[i] for i in ids),'memrl_correct',sum(m[i] for i in ids))
print({k:len(v) for k,v in groups.items()})
print('category,total,region_sr,memrl_sr,diff_pp,memrl_only,region_only')
for cat in sorted({meta[i]['category'] for i in ids}):
    z=[i for i in ids if meta[i]['category']==cat]
    rc=sum(r[i] for i in z); mc=sum(m[i] for i in z)
    print(cat,len(z),100*rc/len(z),100*mc/len(z),100*(rc-mc)/len(z),
          sum(m[i] and not r[i] for i in z),sum(r[i] and not m[i] for i in z),sep=',')
