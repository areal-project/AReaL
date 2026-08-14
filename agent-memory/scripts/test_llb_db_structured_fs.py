#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1] if Path(__file__).parent.name=='scripts' else Path('/storage/openpsi/users/yl/agent-memory/MemRL')
sys.path.insert(0,str(ROOT))
from memrl.lifelongbench_eval.db_failure_summary import db_signature,classify_failure_text,build_structured_db_summary

skills=['select','subquery_nested','group_by_single_column','having_single_condition_with_aggregate','order_by_single_column','limit_and_offset']
sig=db_signature(skills)
assert sig[0]=='select' and sig[1]=='subquery' and 'group_having' in sig[2] and 'ordering' in sig[2]
texts=[
'''FAILURE_MODE: wrong answer\nMISTAKES:\n- The supplier rows used WHERE instead of HAVING for MAX(restock_qty).\n- ORDER BY quantity descending was omitted.\nFIXES:\n- Filter grouped suppliers in HAVING after GROUP BY.\n- Apply ORDER BY before OFFSET and LIMIT.''',
'''FAILURE_MODE: wrong answer\nMISTAKES:\n- The project query compared its group average against the wrong global average subquery.\n- Pagination was applied before ordering.\nFIXES:\n- Compute the overall average in a scalar subquery.\n- ORDER BY first, then LIMIT/OFFSET.''',
]
summary=build_structured_db_summary(texts,skills)
assert 'supplier' not in summary.lower() and 'restock_qty' not in summary.lower() and 'project' not in summary.lower()
assert 'HAVING' in summary and 'subquery scope' in summary and 'ORDER BY' in summary
assert 'comma-separated tuple' in summary
print(summary)
print('LLB_DB_STRUCTURED_FS_TESTS_OK')
