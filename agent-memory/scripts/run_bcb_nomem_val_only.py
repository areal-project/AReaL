#!/usr/bin/env python3
"""Run only the standard 342-task BCB validation split with no memory."""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def args():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--split',default='instruct');p.add_argument('--subset',default='full')
    p.add_argument('--epochs',type=int,default=1);p.add_argument('--output_dir',required=True)
    p.add_argument('--split_file',default=None);p.add_argument('--data_path',default=None);p.add_argument('--seed',type=int,default=42);p.add_argument('--train_ratio',type=float,default=.7)
    p.add_argument('--eval_timeout',type=float,default=240);p.add_argument('--untrusted_hard_timeout',type=float,default=300)
    p.add_argument('--checkpoint_interval',type=int,default=100);p.add_argument('--max_checkpoints',type=int,default=3)
    p.add_argument('--temperature',type=float,default=None);p.add_argument('--max_tokens',type=int,default=None);p.add_argument('--bcb_repo',default=str(ROOT/'3rdparty/bigcodebench-main'))
    return p.parse_args()


def main():
    a=args()
    from memrl.configs.config import MempConfig
    from memrl.providers.llm import OpenAILLM
    from memrl.run.bcb_runner import BCBRunner, BCBSelection
    from memrl.bigcodebench_eval.task_wrappers import load_bcb_data, split_dataset
    cfg=MempConfig.from_yaml(a.config)
    split_file=a.split_file or str(ROOT/'configs/bigcodebench/splits/full_seed42.json')
    run_dir=Path(a.output_dir)/'bigcodebench_eval'/f'{a.split}_{a.subset}'/'nomemory_val'/time.strftime('%Y%m%d_%H%M%S')
    run_dir.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    llm=OpenAILLM(api_key=cfg.llm.api_key,base_url=cfg.llm.base_url,model=cfg.llm.model,
        default_temperature=cfg.llm.temperature if a.temperature is None else a.temperature,
        default_max_tokens=cfg.llm.max_tokens if a.max_tokens is None else a.max_tokens,
        token_log_dir=str(ROOT/'logs/bcb_nomem_val'))
    sel=BCBSelection(subset=a.subset,split=a.split,train_ratio=a.train_ratio,seed=a.seed,split_file=split_file,data_path=a.data_path)
    r=BCBRunner(root=ROOT,selection=sel,llm=llm,memory_service=None,output_dir=str(run_dir),model_name=cfg.llm.model,
        num_epochs=1,run_validation=True,temperature=cfg.llm.temperature if a.temperature is None else a.temperature,
        max_tokens=cfg.llm.max_tokens if a.max_tokens is None else a.max_tokens,retrieve_k=0,bcb_repo=a.bcb_repo,
        eval_timeout_s=a.eval_timeout,untrusted_hard_timeout_s=a.untrusted_hard_timeout,batch_size=int(cfg.experiment.batch_size))
    r._problems=load_bcb_data(subset=a.subset,data_path=a.data_path)
    r._train_ids,r._val_ids=split_dataset(r._problems,train_ratio=a.train_ratio,seed=a.seed,split_file=split_file)
    logging.info('No-memory standard val only: n=%d (train split n=%d, not executed)',len(r._val_ids),len(r._train_ids))
    result=r._run_phase(epoch=1,phase='val',task_ids=r._val_ids,epoch_dir=str(run_dir/'epoch1'),update_memory=False)
    (run_dir/'summary.json').write_text(json.dumps({'standard_heldout_val':True,'train_executed':False,'result':result},indent=2))
    logging.info('DONE no-memory val: %d/%d = %.2f%%',result['pass'],result['total'],100*result['pass']/result['total'])

if __name__=='__main__': main()
