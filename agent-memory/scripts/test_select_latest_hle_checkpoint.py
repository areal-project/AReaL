#!/usr/bin/env python3
import json,subprocess,tempfile
from pathlib import Path
script=Path(__file__).with_name('select_latest_hle_checkpoint.py')
with tempfile.TemporaryDirectory() as td:
 root=Path(td)
 def make(name,section,batch,ids,pending,mtime):
  p=root/name;(p/'local_cache').mkdir(parents=True);(p/'cube').mkdir();(p/'snapshot_meta.json').write_text('{}')
  (p/'local_cache/cum_state.json').write_text(json.dumps({'next_section':section,'next_batch':batch,'batch_all_recs':[{'id':f'q{i}','question':str(i),'correct':False} for i in range(ids)],'pending_task_ids':[f'p{i}' for i in range(pending)]}))
  import os;os.utime(p,(mtime,mtime));return p
 old=make('1_b59',1,60,1849,72,100)
 pending=make('1_pending',1,60,1890,8,90)
 infra=make('1_infra_r1',1,79,1880,2,110)
 out=subprocess.check_output(['python3',str(script),'--glob',str(root/'*'),'--max-no-id','0'],text=True).strip()
 assert out==str(pending),(out,pending)
 print('OK: selector chooses furthest unique-completed state, including pending/infra names')
