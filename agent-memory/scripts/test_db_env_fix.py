#!/usr/bin/env python3
"""Quick sanity test for the LocalDBContainer session compat fix.

Runs a small subset of DB Bench tasks (10 MD5 + 10 DIRECT) end-to-end through
the patched LocalDBContainer to verify:
1. _init_session_compat() works without errors
2. group_concat_max_len is correctly set to 1024
3. ONLY_FULL_GROUP_BY sql_mode is active
4. MD5 validation still passes
5. DIRECT type queries still work
"""

import json
import os
import re
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "3rdparty" / "LifelongAgentBench"))

from memrl.apptainer.patch import patch_llb_containers
patch_llb_containers()

from memrl.apptainer.db_container import LocalDBContainer


def main():
    db_train_path = project_root / "data" / "llb" / "db_train.json"
    if not db_train_path.exists():
        print(f"ERROR: Dataset not found: {db_train_path}")
        sys.exit(1)

    with open(db_train_path) as f:
        dataset = json.load(f)

    md5_samples = []
    direct_samples = []
    for key, entry in dataset.items():
        if entry["answer_info"].get("md5") is not None:
            md5_samples.append((key, entry))
        else:
            direct_samples.append((key, entry))

    test_md5 = md5_samples[:10]
    test_direct = direct_samples[:10]

    print(f"Dataset: {len(dataset)} total ({len(md5_samples)} MD5, {len(direct_samples)} DIRECT)")
    print(f"Testing: {len(test_md5)} MD5 + {len(test_direct)} DIRECT = {len(test_md5) + len(test_direct)} samples")
    print()

    # Start container
    print("[1/5] Starting LocalDBContainer...")
    container = LocalDBContainer()
    print(f"  ✓ mysqld on port {container.port}")

    # Verify session variables
    print("[2/5] Verifying session variables after connect...")
    container.conn.reconnect()
    cursor = container.conn.cursor()
    cursor.execute("SHOW VARIABLES LIKE 'group_concat_max_len'")
    gcml = cursor.fetchall()
    cursor.execute("SELECT @@sql_mode")
    sql_mode = cursor.fetchall()

    # After reconnect, session vars are reset. Call execute() which re-inits.
    container.execute("SELECT 1")
    cursor = container.conn.cursor()
    cursor.execute("SHOW VARIABLES LIKE 'group_concat_max_len'")
    gcml_after = cursor.fetchall()
    cursor.execute("SELECT @@sql_mode")
    sql_mode_after = cursor.fetchall()

    print(f"  group_concat_max_len (after execute): {gcml_after[0][1] if gcml_after else '?'}")
    print(f"  sql_mode (after execute): {sql_mode_after[0][0] if sql_mode_after else '?'}")

    gcml_val = str(gcml_after[0][1]) if gcml_after else ""
    if gcml_val == "1024":
        print("  ✓ group_concat_max_len = 1024 (MySQL 8 parity)")
    else:
        print(f"  ✗ group_concat_max_len = {gcml_val} (EXPECTED 1024!)")
        container.delete()
        sys.exit(1)

    mode_val = str(sql_mode_after[0][0]) if sql_mode_after else ""
    if "ONLY_FULL_GROUP_BY" in mode_val:
        print("  ✓ ONLY_FULL_GROUP_BY is active")
    else:
        print(f"  ✗ ONLY_FULL_GROUP_BY missing! sql_mode={mode_val}")
        container.delete()
        sys.exit(1)

    # Test MD5 samples
    print(f"\n[3/5] Testing {len(test_md5)} MD5-type samples...")
    md5_pass = 0
    md5_fail = 0
    md5_errors = []

    for key, entry in test_md5:
        table_info = entry["table_info"]
        table_name = table_info["name"]
        db_name = table_info["name"]
        column_info_list = table_info["column_info_list"]
        row_list = table_info["row_list"]
        expected_md5 = entry["answer_info"]["md5"]
        ground_truth_sql = entry["answer_info"]["sql"].strip()

        try:
            # Build init SQL (same as DBBench._build_init_sql)
            column_str = ",".join([f"`{c['name']}` {c['type']}" for c in column_info_list])
            column_name_str = ",".join([f"`{c['name']}`" for c in column_info_list])

            item_list = []
            for row in row_list:
                vals = []
                for v in row:
                    if isinstance(v, str):
                        v = v.replace("'", "''")
                    vals.append(f"'{v}'")
                item_list.append(f"({','.join(vals)})")
            item_str = ",".join(item_list)

            init_sql = (
                f"CREATE DATABASE IF NOT EXISTS `{db_name}`;\n"
                f"USE `{db_name}`;\n"
                f"CREATE TABLE IF NOT EXISTS `{table_name}` ({column_str});\n"
                f"INSERT INTO `{table_name}` ({column_name_str}) VALUES {item_str};\n"
                f"COMMIT;\n"
            )

            container.execute(init_sql)

            # Execute ground truth SQL (INSERT/UPDATE/DELETE)
            container.execute(ground_truth_sql, db_name)

            # Compute MD5
            md5_col_str = ",".join([f"`{c['name']}`" for c in column_info_list])
            md5_query = (
                f"SELECT md5(group_concat(rowhash ORDER BY rowhash)) AS hash "
                f"FROM (SELECT SUBSTRING(MD5(CONCAT_WS(',', {md5_col_str})), 1, 5) AS rowhash "
                f"FROM `{table_name}`) AS sub"
            )
            result = container.execute(md5_query, db_name)
            answer_match = re.search(r"\('?(.*?)'?,\)", result)
            computed_md5 = answer_match.group(1) if answer_match else result

            if computed_md5 == expected_md5:
                md5_pass += 1
            else:
                md5_fail += 1
                md5_errors.append(f"  key={key}: expected={expected_md5}, got={computed_md5}")

            # Cleanup
            container.execute(f"DROP DATABASE IF EXISTS `{db_name}`")

        except Exception as e:
            md5_fail += 1
            md5_errors.append(f"  key={key}: ERROR: {str(e)[:100]}")
            try:
                container.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
            except:
                pass

    print(f"  MD5 results: {md5_pass} pass, {md5_fail} fail")
    for err in md5_errors:
        print(err)

    # Test DIRECT samples
    print(f"\n[4/5] Testing {len(test_direct)} DIRECT-type samples...")
    direct_pass = 0
    direct_fail = 0
    direct_errors = []

    for key, entry in test_direct:
        table_info = entry["table_info"]
        table_name = table_info["name"]
        db_name = table_info["name"]
        column_info_list = table_info["column_info_list"]
        row_list = table_info["row_list"]
        ground_truth_sql = entry["answer_info"]["sql"].strip()
        expected_direct = entry["answer_info"]["direct"]

        try:
            column_str = ",".join([f"`{c['name']}` {c['type']}" for c in column_info_list])
            column_name_str = ",".join([f"`{c['name']}`" for c in column_info_list])

            item_list = []
            for row in row_list:
                vals = []
                for v in row:
                    if isinstance(v, str):
                        v = v.replace("'", "''")
                    vals.append(f"'{v}'")
                item_list.append(f"({','.join(vals)})")
            item_str = ",".join(item_list)

            init_sql = (
                f"CREATE DATABASE IF NOT EXISTS `{db_name}`;\n"
                f"USE `{db_name}`;\n"
                f"CREATE TABLE IF NOT EXISTS `{table_name}` ({column_str});\n"
                f"INSERT INTO `{table_name}` ({column_name_str}) VALUES {item_str};\n"
                f"COMMIT;\n"
            )
            container.execute(init_sql)

            # Execute ground truth SQL
            result = container.execute(ground_truth_sql, db_name)

            # Basic check: result should be parseable and non-error
            if result.startswith("[(") or result == "[]":
                direct_pass += 1
            elif "Error" in result or "error" in result:
                direct_fail += 1
                direct_errors.append(f"  key={key}: SQL error: {result[:100]}")
            else:
                direct_pass += 1  # Non-standard but not an error

            container.execute(f"DROP DATABASE IF EXISTS `{db_name}`")

        except Exception as e:
            direct_fail += 1
            direct_errors.append(f"  key={key}: EXCEPTION: {str(e)[:100]}")
            try:
                container.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
            except:
                pass

    print(f"  DIRECT results: {direct_pass} pass, {direct_fail} fail")
    for err in direct_errors:
        print(err)

    # Summary
    print(f"\n[5/5] Summary")
    print("=" * 50)
    total_pass = md5_pass + direct_pass
    total_fail = md5_fail + direct_fail
    total = total_pass + total_fail
    print(f"  Total: {total_pass}/{total} passed ({100*total_pass/total:.1f}%)")
    print(f"  MD5:    {md5_pass}/{len(test_md5)} passed")
    print(f"  DIRECT: {direct_pass}/{len(test_direct)} passed")

    if total_fail == 0:
        print("\n  ✓ ALL TESTS PASSED — environment fix is working correctly")
    else:
        print(f"\n  ✗ {total_fail} FAILURES — investigate errors above")

    container.delete()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
