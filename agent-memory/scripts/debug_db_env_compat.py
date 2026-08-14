#!/usr/bin/env python3
"""Diagnostic script: verify MariaDB 10.6 vs MySQL 8 compatibility for LLB DB Bench.

Checks the three highest-risk environment differences identified by code review:
1. group_concat_max_len default (MySQL=1024 vs MariaDB=1048576)
2. Default collation (MySQL=utf8mb4_0900_ai_ci vs MariaDB=utf8mb4_general_ci)
3. sql_mode defaults

Also runs a concrete MD5 validation test with sample data to check if any of these
differences actually produce different results.
"""

import json
import os
import sys
import tempfile
import shutil
import subprocess
import time
import socket
import random

try:
    import mysql.connector
except ImportError:
    print("ERROR: mysql-connector-python not installed")
    sys.exit(1)


def find_mysqld():
    for name in ["mysqld", "mariadbd"]:
        path = shutil.which(name)
        if path:
            return path
    for candidate in ["/usr/libexec/mysqld", "/usr/sbin/mysqld", "/usr/sbin/mariadbd"]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("mysqld not found")


def start_mysqld():
    mysqld_bin = find_mysqld()
    datadir = tempfile.mkdtemp(prefix="llb_dbcheck_")

    version_result = subprocess.run(
        [mysqld_bin, "--version"], capture_output=True, text=True, timeout=10
    )
    version_out = version_result.stdout + version_result.stderr
    is_mariadb = "MariaDB" in version_out
    print(f"[INFO] mysqld binary: {mysqld_bin}")
    print(f"[INFO] Version: {version_out.strip()}")
    print(f"[INFO] Is MariaDB: {is_mariadb}")

    if is_mariadb:
        install_db = shutil.which("mariadb-install-db") or shutil.which("mysql_install_db") or "/usr/bin/mysql_install_db"
        subprocess.run(
            [install_db, f"--datadir={datadir}", "--user=root"],
            capture_output=True, text=True, timeout=60,
        )
    else:
        subprocess.run(
            [mysqld_bin, "--initialize-insecure", f"--datadir={datadir}", "--user=root"],
            capture_output=True, text=True, timeout=60,
        )

    port = 13000 + random.randint(0, 10000)
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(("localhost", port))
                port += random.randint(1, 20)
            except (ConnectionRefusedError, OSError):
                break

    proc = subprocess.Popen(
        [
            mysqld_bin,
            f"--datadir={datadir}",
            f"--port={port}",
            "--user=root",
            "--skip-grant-tables",
            f"--socket={datadir}/mysql.sock",
            f"--pid-file={datadir}/mysqld.pid",
            "--bind-address=127.0.0.1",
            "--innodb-buffer-pool-size=32M",
            "--innodb-log-file-size=16M",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)

    if proc.poll() is not None:
        stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
        raise RuntimeError(f"mysqld failed to start: {stderr[-1000:]}")

    for retry in range(20):
        try:
            conn = mysql.connector.connect(host="127.0.0.1", user="root", port=port)
            return conn, proc, datadir, port, is_mariadb
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Could not connect to mysqld on port {port}")


def check_session_variables(conn):
    """Check critical session variables that differ between MySQL 8 and MariaDB 10.6."""
    cursor = conn.cursor()

    checks = [
        "group_concat_max_len",
        "collation_server",
        "collation_database",
        "collation_connection",
        "character_set_server",
        "character_set_database",
        "character_set_connection",
        "sql_mode",
        "innodb_strict_mode",
    ]

    print("\n" + "=" * 60)
    print("CHECK 1: Critical Session/Global Variables")
    print("=" * 60)

    results = {}
    for var in checks:
        cursor.execute(f"SHOW VARIABLES LIKE '{var}'")
        row = cursor.fetchall()
        if row:
            val = row[0][1]
            results[var] = val
            flag = ""
            if var == "group_concat_max_len" and str(val) != "1024":
                flag = " ⚠️  MySQL 8 default is 1024!"
            if var == "collation_server" and "0900" not in str(val):
                flag = " ⚠️  MySQL 8 default is utf8mb4_0900_ai_ci!"
            print(f"  {var} = {val}{flag}")
        else:
            results[var] = None
            print(f"  {var} = (not found)")

    return results


def check_md5_group_concat_behavior(conn):
    """Test if GROUP_CONCAT truncation or collation ordering affects MD5 validation."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("CHECK 2: MD5 + GROUP_CONCAT Behavior Test")
    print("=" * 60)

    cursor.execute("CREATE DATABASE IF NOT EXISTS `__dbcheck_test`")
    cursor.fetchall()
    cursor.execute("USE `__dbcheck_test`")
    cursor.fetchall()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `test_table` (
            `id` INT,
            `name` VARCHAR(100),
            `value` DECIMAL(10,2),
            `description` TEXT
        )
    """)
    cursor.fetchall()

    cursor.execute("TRUNCATE TABLE `test_table`")
    cursor.fetchall()

    rows = [
        (1, "Alice", 100.50, "First entry"),
        (2, "Bob", 200.75, "Second entry"),
        (3, "Charlie", 50.00, "Third entry"),
        (4, "Diana", 300.25, "Fourth entry"),
        (5, "Eve", 150.10, "Fifth entry"),
        (6, None, 0.00, None),
        (7, "Grace", 99.99, "Seventh entry with special chars: O'Brien"),
        (8, "Héloïse", 42.42, "Unicode name"),
        (9, "Ivan", 1000000.01, "Large number"),
        (10, "Jenny", -50.50, "Negative value"),
    ]
    for r in rows:
        cursor.execute(
            "INSERT INTO `test_table` (`id`, `name`, `value`, `description`) VALUES (%s, %s, %s, %s)",
            r
        )
    conn.commit()

    md5_query = """
        SELECT md5(group_concat(rowhash ORDER BY rowhash)) AS hash
        FROM (
            SELECT SUBSTRING(MD5(CONCAT_WS(',', `id`, `name`, `value`, `description`)), 1, 5) AS rowhash
            FROM `test_table`
        ) AS sub
    """

    cursor.execute(md5_query)
    result_default = cursor.fetchall()
    print(f"  MD5 (default group_concat_max_len): {result_default}")

    cursor.execute("SET SESSION group_concat_max_len = 1024")
    cursor.fetchall()
    cursor.execute(md5_query)
    result_1024 = cursor.fetchall()
    print(f"  MD5 (group_concat_max_len=1024):    {result_1024}")

    cursor.execute("SET SESSION group_concat_max_len = 1048576")
    cursor.fetchall()
    cursor.execute(md5_query)
    result_large = cursor.fetchall()
    print(f"  MD5 (group_concat_max_len=1048576): {result_large}")

    if result_default == result_1024 == result_large:
        print("  ✅ All three match — group_concat_max_len does NOT affect this test case")
    else:
        print("  ❌ MISMATCH detected! group_concat_max_len affects MD5 result!")

    print("\n  --- Testing with larger data (to trigger truncation) ---")
    cursor.execute("TRUNCATE TABLE `test_table`")
    cursor.fetchall()
    for i in range(200):
        cursor.execute(
            "INSERT INTO `test_table` (`id`, `name`, `value`, `description`) VALUES (%s, %s, %s, %s)",
            (i, f"User_{i}_{'x' * 50}", float(i) * 1.23, f"Description for user {i} with padding: {'y' * 100}")
        )
    conn.commit()

    cursor.execute("SET SESSION group_concat_max_len = 1024")
    cursor.fetchall()
    cursor.execute(md5_query)
    result_1024_large = cursor.fetchall()
    print(f"  MD5 (200 rows, group_concat_max_len=1024):    {result_1024_large}")

    cursor.execute("SET SESSION group_concat_max_len = 1048576")
    cursor.fetchall()
    cursor.execute(md5_query)
    result_large_large = cursor.fetchall()
    print(f"  MD5 (200 rows, group_concat_max_len=1048576): {result_large_large}")

    if result_1024_large == result_large_large:
        print("  ✅ Match — even with 200 rows, no truncation issue")
    else:
        print("  ❌ MISMATCH! GROUP_CONCAT truncation at 1024 causes different MD5!")
        print("     This is a CONFIRMED environment bug affecting MD5-type task validation!")


def check_collation_ordering(conn):
    """Test if collation differences affect ORDER BY in GROUP_CONCAT."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("CHECK 3: Collation-dependent ORDER BY in GROUP_CONCAT")
    print("=" * 60)

    cursor.execute("USE `__dbcheck_test`")
    cursor.fetchall()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `collation_test` (
            `val` VARCHAR(10)
        )
    """)
    cursor.fetchall()
    cursor.execute("TRUNCATE TABLE `collation_test`")
    cursor.fetchall()

    test_vals = ['abc', 'ABC', 'def', 'DEF', 'Abc', 'aBc', '123', 'aBC', 'ABc']
    for v in test_vals:
        cursor.execute("INSERT INTO `collation_test` VALUES (%s)", (v,))
    conn.commit()

    cursor.execute("SET SESSION group_concat_max_len = 1048576")
    cursor.fetchall()

    cursor.execute("SELECT GROUP_CONCAT(val ORDER BY val) FROM `collation_test`")
    result_default = cursor.fetchall()[0][0]
    print(f"  ORDER BY val (default collation): {result_default}")

    cursor.execute("SELECT GROUP_CONCAT(val ORDER BY val COLLATE utf8mb4_bin) FROM `collation_test`")
    result_bin = cursor.fetchall()[0][0]
    print(f"  ORDER BY val COLLATE utf8mb4_bin:  {result_bin}")

    try:
        cursor.execute("SELECT GROUP_CONCAT(val ORDER BY val COLLATE utf8mb4_0900_ai_ci) FROM `collation_test`")
        result_0900 = cursor.fetchall()[0][0]
        print(f"  ORDER BY val COLLATE utf8mb4_0900_ai_ci: {result_0900}")
    except Exception as e:
        print(f"  ORDER BY val COLLATE utf8mb4_0900_ai_ci: NOT SUPPORTED ({e})")
        result_0900 = None

    try:
        cursor.execute("SELECT GROUP_CONCAT(val ORDER BY val COLLATE utf8mb4_general_ci) FROM `collation_test`")
        result_general = cursor.fetchall()[0][0]
        print(f"  ORDER BY val COLLATE utf8mb4_general_ci: {result_general}")
    except Exception as e:
        print(f"  ORDER BY val COLLATE utf8mb4_general_ci: NOT SUPPORTED ({e})")
        result_general = None

    if result_0900 and result_general and result_0900 != result_general:
        print("  ⚠️  utf8mb4_0900_ai_ci and utf8mb4_general_ci produce DIFFERENT ordering!")
    elif result_0900 and result_general:
        print("  ✅ Same ordering for both collations on this test data")

    print(f"\n  Note: MD5 rowhash is hex [0-9a-f], so collation usually doesn't matter for MD5 queries.")
    print(f"  But for DIRECT type tasks where agent output is compared as string, collation")
    print(f"  affects ORDER BY in the agent's SQL queries → potentially different result sets.")


def check_sql_mode(conn):
    """Check sql_mode differences."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("CHECK 4: sql_mode")
    print("=" * 60)

    cursor.execute("SELECT @@sql_mode")
    result = cursor.fetchall()
    current_mode = result[0][0] if result else "(unknown)"
    print(f"  Current sql_mode: {current_mode}")

    mysql8_default = "ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION"
    print(f"  MySQL 8 default:  {mysql8_default}")

    current_set = set(current_mode.split(",")) if current_mode else set()
    mysql8_set = set(mysql8_default.split(","))

    missing = mysql8_set - current_set
    extra = current_set - mysql8_set

    if missing:
        print(f"  ⚠️  Missing vs MySQL 8: {', '.join(sorted(missing))}")
    if extra:
        print(f"  ⚠️  Extra vs MySQL 8:   {', '.join(sorted(extra))}")
    if not missing and not extra:
        print(f"  ✅ sql_mode matches MySQL 8 defaults")


def check_fetchall_format(conn):
    """Check if cursor.fetchall() returns identical string format."""
    cursor = conn.cursor()

    print("\n" + "=" * 60)
    print("CHECK 5: cursor.fetchall() String Format")
    print("=" * 60)

    cursor.execute("USE `__dbcheck_test`")
    cursor.fetchall()

    cursor.execute("SELECT * FROM `test_table` ORDER BY `id` LIMIT 3")
    result = cursor.fetchall()
    result_str = str(result)
    print(f"  str(fetchall()) = {result_str[:200]}...")
    print(f"  Type of first row element types:")
    if result:
        for i, elem in enumerate(result[0]):
            print(f"    Column {i}: type={type(elem).__name__}, value={repr(elem)}")


def check_real_db_bench_sample(conn):
    """Test with an actual DB Bench sample from the dataset."""
    print("\n" + "=" * 60)
    print("CHECK 6: Real DB Bench Sample Validation")
    print("=" * 60)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_train_path = os.path.join(project_root, "data", "llb", "db_train.json")

    if not os.path.exists(db_train_path):
        print(f"  ⚠️  Dataset not found: {db_train_path}")
        return

    with open(db_train_path) as f:
        dataset = json.load(f)

    cursor = conn.cursor()

    md5_samples = []
    direct_samples = []
    for key, entry in dataset.items():
        answer_info = entry["answer_info"]
        if answer_info.get("md5") is not None:
            md5_samples.append((key, entry))
        else:
            direct_samples.append((key, entry))

    print(f"  Dataset: {len(dataset)} total, {len(md5_samples)} MD5, {len(direct_samples)} DIRECT")

    md5_match = 0
    md5_mismatch = 0
    md5_errors = []

    for key, entry in md5_samples[:30]:
        table_info = entry["table_info"]
        table_name = table_info["name"]
        db_name = table_info["name"]
        column_info_list = table_info["column_info_list"]
        row_list = table_info["row_list"]
        expected_md5 = entry["answer_info"]["md5"]
        ground_truth_sql = entry["answer_info"]["sql"].strip()

        try:
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS `{db_name}`")
            cursor.fetchall()
            cursor.execute(f"USE `{db_name}`")
            cursor.fetchall()

            column_str = ",".join([f"`{c['name']}` {c['type']}" for c in column_info_list])
            cursor.execute(f"CREATE TABLE IF NOT EXISTS `{table_name}` ({column_str})")
            cursor.fetchall()

            column_name_str = ",".join([f"`{c['name']}`" for c in column_info_list])
            for row in row_list:
                placeholders = ",".join(["%s"] * len(row))
                cursor.execute(f"INSERT INTO `{table_name}` ({column_name_str}) VALUES ({placeholders})", row)
            conn.commit()

            cursor.execute(ground_truth_sql)
            cursor.fetchall()
            conn.commit()

            md5_col_str = ",".join([f"`{c['name']}`" for c in column_info_list])
            md5_query = (
                f"SELECT md5(group_concat(rowhash ORDER BY rowhash)) AS hash "
                f"FROM (SELECT SUBSTRING(MD5(CONCAT_WS(',', {md5_col_str})), 1, 5) AS rowhash "
                f"FROM `{table_name}`) AS sub"
            )
            cursor.execute(md5_query)
            result = cursor.fetchall()
            import re
            answer_match = re.search(r"\('?(.*?)'?,\)", str(result))
            computed_md5 = answer_match.group(1) if answer_match else str(result)

            if computed_md5 == expected_md5:
                md5_match += 1
            else:
                md5_mismatch += 1
                md5_errors.append({
                    "key": key,
                    "expected": expected_md5,
                    "got": computed_md5,
                    "sql": ground_truth_sql[:100],
                })

            cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
            cursor.fetchall()

        except Exception as e:
            md5_errors.append({"key": key, "error": str(e)[:200]})
            try:
                cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
                cursor.fetchall()
            except:
                pass

    print(f"\n  MD5 validation (first 30 MD5-type samples):")
    print(f"    Match:    {md5_match}")
    print(f"    Mismatch: {md5_mismatch}")
    if md5_errors:
        print(f"    Errors ({len(md5_errors)}):")
        for err in md5_errors[:5]:
            print(f"      {err}")

    if md5_mismatch > 0:
        print(f"\n  ❌ CONFIRMED: MariaDB produces different MD5 for {md5_mismatch} samples!")
        print(f"     This is a real environment bug causing false negatives!")
    elif md5_match > 0 and md5_mismatch == 0:
        print(f"\n  ✅ All {md5_match} tested MD5 samples match expected ground truth")


def main():
    print("=" * 60)
    print("LLB DB Bench — MariaDB 10.6 vs MySQL 8 Compatibility Check")
    print("=" * 60)

    conn, proc, datadir, port, is_mariadb = start_mysqld()
    print(f"[INFO] mysqld started on port {port} (MariaDB={is_mariadb})")

    try:
        vars_result = check_session_variables(conn)
        check_md5_group_concat_behavior(conn)
        check_collation_ordering(conn)
        check_sql_mode(conn)
        check_fetchall_format(conn)
        check_real_db_bench_sample(conn)

        conn.cursor().execute("DROP DATABASE IF EXISTS `__dbcheck_test`")

        print("\n" + "=" * 60)
        print("SUMMARY OF FINDINGS")
        print("=" * 60)

        gcml = vars_result.get("group_concat_max_len", "?")
        collation = vars_result.get("collation_server", "?")
        sql_mode = vars_result.get("sql_mode", "?")

        issues = []
        if str(gcml) != "1024":
            issues.append(f"group_concat_max_len={gcml} (MySQL 8 default=1024)")
        if "0900" not in str(collation):
            issues.append(f"collation_server={collation} (MySQL 8 default=utf8mb4_0900_ai_ci)")
        if "ONLY_FULL_GROUP_BY" not in str(sql_mode):
            issues.append(f"sql_mode missing ONLY_FULL_GROUP_BY")

        if issues:
            print("  POTENTIAL ISSUES:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("  No critical differences detected.")

    finally:
        conn.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except:
            proc.kill()
        shutil.rmtree(datadir, ignore_errors=True)
        print(f"\n[INFO] Cleanup complete.")


if __name__ == "__main__":
    main()
