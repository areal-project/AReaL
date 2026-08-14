"""Container adapters for DB Bench tasks.

Provides two backends:
- LocalDBContainer: starts mysqld directly on host (requires mysql installed)
- ApptainerDBContainer: starts mysqld inside Apptainer instance

The default depends on MEMRL_DB_BACKEND env var and what's available.
"""

import atexit
import logging
import os
import random
import shutil
import socket
import subprocess
import tempfile
import time
import uuid
from typing import Optional

import mysql.connector

logger = logging.getLogger(__name__)


class LocalDBContainer:
    """Run mysqld directly on the host in a temporary data directory.

    Supports both MySQL 8+ (--initialize-insecure) and MariaDB 5.5+ (mysql_install_db).
    """

    port = 13000
    password = "password"

    def __init__(self, image: str = "mysql"):
        self.deleted = False
        self._datadir = tempfile.mkdtemp(prefix="llb_mysql_")
        self._mysqld_bin = self._find_mysqld()
        self._use_password = True  # Whether to connect with password

        p = LocalDBContainer.port + random.randint(0, 10000)
        while self.is_port_open(p):
            p += random.randint(1, 20)
        self.port = p

        self._start_mysqld()
        self._wait_and_connect()

    @staticmethod
    def _find_mysqld() -> str:
        """Find mysqld binary (may not be on PATH)."""
        for name in ["mysqld", "mariadbd"]:
            path = shutil.which(name)
            if path:
                return path
        for candidate in ["/usr/libexec/mysqld", "/usr/sbin/mysqld", "/usr/sbin/mariadbd", "/usr/local/bin/mysqld"]:
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        raise RuntimeError("mysqld not found. Install mysql-server or mariadb-server.")

    def _start_mysqld(self) -> None:
        # Detect MariaDB vs MySQL for initialization
        version_result = subprocess.run(
            [self._mysqld_bin, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version_out = version_result.stdout + version_result.stderr
        is_mariadb = "MariaDB" in version_out
        logger.info("mysqld version: %s (MariaDB=%s)", version_out.strip(), is_mariadb)

        if is_mariadb:
            install_db = (
                shutil.which("mariadb-install-db")
                or shutil.which("mysql_install_db")
                or "/usr/bin/mysql_install_db"
            )
            init_result = subprocess.run(
                [install_db, f"--datadir={self._datadir}", "--user=root"],
                capture_output=True, text=True, timeout=60,
            )
            if init_result.returncode != 0:
                logger.warning("mysql_install_db failed (rc=%d): stdout=%s stderr=%s",
                               init_result.returncode, init_result.stdout[:300], init_result.stderr[:300])
        else:
            init_result = subprocess.run(
                [self._mysqld_bin, "--initialize-insecure", f"--datadir={self._datadir}", "--user=root"],
                capture_output=True, text=True, timeout=60,
            )
            if init_result.returncode != 0:
                logger.warning("mysqld --initialize-insecure failed (rc=%d): %s",
                               init_result.returncode, init_result.stderr[:300])

        self._mysqld_proc = subprocess.Popen(
            [
                self._mysqld_bin,
                f"--datadir={self._datadir}",
                f"--port={self.port}",
                "--user=root",
                "--skip-grant-tables",
                f"--socket={self._datadir}/mysql.sock",
                f"--pid-file={self._datadir}/mysqld.pid",
                f"--bind-address=127.0.0.1",
                "--innodb-buffer-pool-size=32M",
                "--innodb-log-file-size=16M",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        time.sleep(5)

        if self._mysqld_proc.poll() is not None:
            stderr = self._mysqld_proc.stderr.read().decode("utf-8", errors="replace") if self._mysqld_proc.stderr else ""
            logger.error("mysqld exited immediately (rc=%d): %s", self._mysqld_proc.returncode, stderr[-2000:])
            raise RuntimeError(f"mysqld failed to start: {stderr[-2000:]}")

        self._use_password = False
        logger.info("Started local mysqld on port %d, datadir=%s", self.port, self._datadir)

    def _wait_and_connect(self) -> None:
        max_retries = 20
        connect_kwargs = {
            "host": "127.0.0.1",
            "user": "root",
            "port": self.port,
        }
        if self._use_password:
            connect_kwargs["password"] = self.password

        for retry in range(max_retries):
            try:
                self.conn = mysql.connector.connect(**connect_kwargs)
                self._init_session_compat()
                return
            except mysql.connector.errors.OperationalError:
                time.sleep(1)
            except (mysql.connector.InterfaceError, mysql.connector.errors.ProgrammingError):
                if retry > 15:
                    raise
                time.sleep(3)
        raise RuntimeError(f"Could not connect to local MySQL on port {self.port}")

    def _init_session_compat(self) -> None:
        """Align MariaDB session variables with MySQL 8 defaults for result parity."""
        cursor = self.conn.cursor()
        cursor.execute("SET SESSION group_concat_max_len = 1024")
        cursor.execute(
            "SET SESSION sql_mode = "
            "'ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,"
            "NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'"
        )

    def delete(self) -> None:
        if not self.deleted:
            try:
                self.conn.close()
            except Exception:
                pass
            try:
                self._mysqld_proc.terminate()
                self._mysqld_proc.wait(timeout=5)
            except Exception:
                try:
                    self._mysqld_proc.kill()
                except Exception:
                    pass
            shutil.rmtree(self._datadir, ignore_errors=True)
            self.deleted = True
            logger.debug("Stopped local mysqld on port %d", self.port)

    def execute(self, multiple_sql: str, database: Optional[str] = None) -> str:
        self.conn.reconnect()
        self._init_session_compat()
        try:
            cursor = self.conn.cursor()
            if database:
                cursor.execute(f"use `{database}`;")
                cursor.fetchall()
            sql_list = multiple_sql.split(";")
            sql_list = [sql.strip() for sql in sql_list if sql.strip() != ""]
            result = ""
            for sql in sql_list:
                cursor.execute(sql)
                result = str(cursor.fetchall())
                self.conn.commit()
        except Exception as e:
            result = str(e)
        return result

    def is_port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(("localhost", port))
                return True
            except (ConnectionRefusedError, OSError):
                return False

    def __del__(self) -> None:
        try:
            if not self.deleted:
                self.delete()
        except Exception:
            pass


# --- Apptainer backend ---

_active_db_instances: set = set()


def _cleanup_db_instances():
    for name in list(_active_db_instances):
        try:
            subprocess.run(
                ["apptainer", "instance", "stop", name],
                capture_output=True,
                timeout=10,
            )
        except Exception:
            pass
    _active_db_instances.clear()


atexit.register(_cleanup_db_instances)

DEFAULT_MYSQL_IMAGE = "/storage/openpsi/images/mysql.sif"


class ApptainerDBContainer:
    """Apptainer-based MySQL container for DB Bench tasks."""

    port = 13000
    password = "password"

    def __init__(self, image: str = DEFAULT_MYSQL_IMAGE):
        self.deleted = False
        self.image = image
        self.instance_name = f"mysql_{uuid.uuid4().hex[:8]}"

        p = ApptainerDBContainer.port + random.randint(0, 10000)
        while self.is_port_open(p):
            p += random.randint(1, 20)
        self.port = p

        self._start_mysql_instance()
        self._wait_and_connect()

    def _start_mysql_instance(self) -> None:
        cmd = [
            "apptainer", "instance", "start",
            "--writable-tmpfs",
            "--no-home",
            "--containall",
            self.image,
            self.instance_name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start MySQL instance '{self.instance_name}': "
                f"{result.stderr.strip()}"
            )
        _active_db_instances.add(self.instance_name)

        init_and_start = (
            "mysqld --initialize-insecure --user=root --datadir=/var/lib/mysql 2>/dev/null; "
            f"mysqld --user=root --port={self.port} --datadir=/var/lib/mysql "
            "--skip-grant-tables &"
        )
        subprocess.run(
            [
                "apptainer", "exec",
                f"instance://{self.instance_name}",
                "bash", "-c", init_and_start,
            ],
            capture_output=True, text=True, timeout=30,
        )
        time.sleep(3)

        # Set root password
        set_pass_sql = (
            f"FLUSH PRIVILEGES; "
            f"ALTER USER 'root'@'localhost' IDENTIFIED BY '{self.password}'; "
            f"FLUSH PRIVILEGES;"
        )
        subprocess.run(
            [
                "apptainer", "exec",
                f"instance://{self.instance_name}",
                "bash", "-c",
                f'mysql --user=root --port={self.port} -e "{set_pass_sql}"',
            ],
            capture_output=True, text=True, timeout=15,
        )
        time.sleep(1)
        logger.debug("Started MySQL in Apptainer: %s (port %d)", self.instance_name, self.port)

    def _wait_and_connect(self) -> None:
        max_retries = 20
        for retry in range(max_retries):
            try:
                self.conn = mysql.connector.connect(
                    host="127.0.0.1",
                    user="root",
                    password=self.password,
                    port=self.port,
                    pool_reset_session=True,
                )
                logger.debug("Connected to MySQL on port %d", self.port)
                return
            except mysql.connector.errors.OperationalError:
                time.sleep(1)
            except mysql.connector.InterfaceError:
                if retry > 15:
                    raise
                time.sleep(3)

        raise RuntimeError(
            f"Could not connect to MySQL on port {self.port} after {max_retries} retries"
        )

    def delete(self) -> None:
        if not self.deleted:
            try:
                self.conn.close()
            except Exception:
                pass
            cmd = ["apptainer", "instance", "stop", self.instance_name]
            subprocess.run(cmd, capture_output=True, timeout=10)
            _active_db_instances.discard(self.instance_name)
            self.deleted = True
            logger.debug("Stopped MySQL instance: %s", self.instance_name)

    def execute(self, multiple_sql: str, database: Optional[str] = None) -> str:
        self.conn.reconnect()
        try:
            cursor = self.conn.cursor()
            if database:
                cursor.execute(f"use `{database}`;")
                cursor.fetchall()
            sql_list = multiple_sql.split(";")
            sql_list = [sql.strip() for sql in sql_list if sql.strip() != ""]
            result = ""
            for sql in sql_list:
                cursor.execute(sql)
                result = str(cursor.fetchall())
                self.conn.commit()
        except Exception as e:
            result = str(e)
        return result

    def is_port_open(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.connect(("localhost", port))
                return True
            except (ConnectionRefusedError, OSError):
                return False

    def __del__(self) -> None:
        try:
            if not self.deleted:
                self.delete()
        except Exception:
            pass
