#!/usr/bin/env python3
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / 'memrl/providers/llm.py'

def worker(root, queue, hold):
    os.environ['MEMRL_LLM_GLOBAL_MAX_INFLIGHT']='2'
    os.environ['MEMRL_LLM_INFLIGHT_DIR']=root
    os.environ['MEMRL_LLM_INFLIGHT_KEY']='test-shared'
    from memrl.providers import llm as mod
    started=time.monotonic()
    h=mod._SharedInflightLimiter.acquire('gemini-3.5-flash')
    acquired=time.monotonic()
    queue.put((started,acquired))
    time.sleep(hold)
    mod._SharedInflightLimiter.release(h)

if __name__=='__main__':
    with tempfile.TemporaryDirectory() as root:
        q=mp.Queue()
        ps=[mp.Process(target=worker,args=(root,q,0.35)) for _ in range(3)]
        for p in ps:p.start()
        rows=[q.get(timeout=3) for _ in ps]
        for p in ps:p.join(3)
        waits=sorted(acq-start for start,acq in rows)
        assert waits[0] < .15 and waits[1] < .15, waits
        assert waits[2] >= .25, waits
        print('OK: cross-process LLM in-flight limit=2 blocks the third request',waits)
