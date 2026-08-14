#!/usr/bin/env python3
import argparse, collections, json, shutil, time
from pathlib import Path
import pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--checkpoint',required=True)
ap.add_argument('--dataset',required=True)
a=ap.parse_args()
dst=Path(a.checkpoint); state_path=dst/'local_cache/cum_state.json'
backup=state_path.with_suffix('.json.pre_idfix')
if not backup.exists(): shutil.copy2(state_path,backup)
state=json.loads(state_path.read_text())
df=pd.read_parquet(a.dataset)
q2ids=collections.defaultdict(list)
for _,row in df.iterrows(): q2ids[str(row['question'])].append(str(row['id']))
records=state.get('batch_all_recs',[]) or []
# Group uniquely mappable legacy and ID-bearing records by stable task key.
groups=collections.OrderedDict(); stats=collections.Counter()
for pos,rec in enumerate(records):
    if not isinstance(rec,dict): continue
    item=dict(rec); has_id=item.get('id') is not None
    if has_id:
        key=('id',str(item['id']))
    else:
        ids=q2ids.get(str(item.get('question','')),[])
        if len(ids)==1:
            item['id']=ids[0]; key=('id',ids[0]); stats['mapped']+=1
        elif len(ids)>1:
            key=('ambiguous_question',str(item.get('question',''))); stats['ambiguous_kept']+=1
        else:
            key=('missing_question',str(item.get('question',''))); stats['missing']+=1
    groups.setdefault(key,[]).append((pos,item,has_id))
# Independent-epoch policy: prefer the latest ID-bearing CheckpointV2 result;
# otherwise keep the latest legacy result. any-success is cumulative and invalid here.
dedup=[]
for key,items in groups.items():
    id_items=[z for z in items if z[2]]
    chosen=(id_items[-1] if id_items else items[-1])[1]
    if len(items)>1: stats['duplicates_dropped'] += len(items)-1
    dedup.append({'id':chosen.get('id'),'question':str(chosen.get('question','')),'correct':bool(chosen.get('correct',False))})
completed={str(r['id']) for r in dedup if r.get('id') is not None}
state['batch_all_recs']=dedup
for key in ['pending_task_ids','terminal_incorrect_task_ids','infrastructure_deferred_task_ids']:
    state[key]=sorted({str(x) for x in state.get(key,[]) if x is not None and str(x) not in completed})
state['legacy_task_id_migration']={'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'source_records':len(records),'deduped_records':len(dedup),**stats}
state_path.write_text(json.dumps(state,ensure_ascii=False,indent=2))
meta_path=dst/'snapshot_meta.json'
meta=json.loads(meta_path.read_text());meta['checkpoint_id']=dst.name;meta['legacy_task_id_fixed']=True;meta['legacy_task_id_fix_records']=len(dedup);meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2))
correct=sum(bool(r['correct']) for r in dedup)
print(json.dumps({'records':len(dedup),'unique_ids':len(completed),'no_id':len(dedup)-len(completed),'correct':correct,'sr':correct/max(1,len(dedup))*100,'pending':len(state['pending_task_ids']),'terminal':len(state['terminal_incorrect_task_ids']),'deferred':len(state['infrastructure_deferred_task_ids']),'stats':dict(stats)},ensure_ascii=False,indent=2))
