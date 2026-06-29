"""OceanBase 数据库集成示例

此脚本演示如何将训练指标持久化到 OceanBase 数据库。

OceanBase 是一个兼容 MySQL 的开源分布式数据库，适合存储大规模训练指标。

使用方法:
    python examples/utils/oceanbase_example.py

环境变量配置:
    OB_HOST: OceanBase 主机地址（默认: 127.0.0.1）
    OB_PORT: OceanBase 端口（默认: 2881）
    OB_USER: 用户名（默认: root@test）
    OB_PASSWORD: 密码（默认: 空）
    OB_DATABASE: 数据库名（默认: test）
"""

import os
from datetime import datetime
from typing import Optional

import pymysql
from pymysql.cursors import DictCursor

from areal.utils.logging import getLogger

logger = getLogger("OceanBaseExample")


class OceanBaseMetricsLogger:
    """OceanBase 指标记录器

    封装与 OceanBase 的连接和指标写入操作。

    Attributes:
        host: 数据库主机地址
        port: 数据库端口
        user: 用户名
        password: 密码
        database: 数据库名
        connection: 数据库连接对象
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 2881,
        user: str = "root@test",
        password: str = "",
        database: str = "test",
    ):
        """初始化 OceanBase 连接

        Args:
            host: 数据库主机地址
            port: 数据库端口
            user: 用户名（格式: user@tenant）
            password: 密码
            database: 数据库名
        """
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection: Optional[pymysql.Connection] = None

    def connect(self) -> None:
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                cursorclass=DictCursor,
                autocommit=True,
            )
            logger.info(
                f"成功连接到 OceanBase: {self.host}:{self.port}/{self.database}"
            )
        except pymysql.Error as e:
            logger.error(f"连接 OceanBase 失败: {e}")
            raise

    def close(self) -> None:
        """关闭数据库连接"""
        if self.connection:
            self.connection.close()
            logger.info("已关闭 OceanBase 连接")

    def create_table(self) -> None:
        """创建训练指标表

        表结构:
            - id: 自增主键
            - experiment_name: 实验名称
            - step: 训练步数
            - loss: 损失值
            - reward: 奖励值（可选）
            - timestamp: 记录时间
        """
        if not self.connection:
            raise RuntimeError("未建立数据库连接，请先调用 connect()")

        create_table_sql = """
        CREATE TABLE IF NOT EXISTS training_metrics (
            id BIGINT AUTO_INCREMENT PRIMARY KEY,
            experiment_name VARCHAR(100) NOT NULL,
            step INT NOT NULL,
            loss FLOAT,
            reward FLOAT,
            timestamp DATETIME NOT NULL,
            INDEX idx_experiment_step (experiment_name, step),
            INDEX idx_timestamp (timestamp)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(create_table_sql)
            logger.info("成功创建表 training_metrics")
        except pymysql.Error as e:
            logger.error(f"创建表失败: {e}")
            raise

    def insert_metric(
        self,
        experiment_name: str,
        step: int,
        loss: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> None:
        """插入单条训练指标

        Args:
            experiment_name: 实验名称
            step: 训练步数
            loss: 损失值
            reward: 奖励值
        """
        if not self.connection:
            raise RuntimeError("未建立数据库连接，请先调用 connect()")

        insert_sql = """
        INSERT INTO training_metrics
        (experiment_name, step, loss, reward, timestamp)
        VALUES (%s, %s, %s, %s, %s)
        """

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    insert_sql,
                    (experiment_name, step, loss, reward, datetime.now()),
                )
            logger.info(
                f"插入指标: {experiment_name} step={step} loss={loss} reward={reward}"
            )
        except pymysql.Error as e:
            logger.error(f"插入指标失败: {e}")
            raise


def main():
    """主函数：演示 OceanBase 集成"""
    # 从环境变量读取配置
    config = {
        "host": os.getenv("OB_HOST", "127.0.0.1"),
        "port": int(os.getenv("OB_PORT", "2881")),
        "user": os.getenv("OB_USER", "root@test"),
        "password": os.getenv("OB_PASSWORD", ""),
        "database": os.getenv("OB_DATABASE", "test"),
    }

    logger.info("=== OceanBase 集成示例 ===")
    logger.info(f"连接配置: {config['host']}:{config['port']}/{config['database']}")

    # 创建指标记录器
    metrics_logger = OceanBaseMetricsLogger(**config)

    try:
        # 1. 连接数据库
        metrics_logger.connect()

        # 2. 创建表
        metrics_logger.create_table()

        # 3. 插入示例数据
        logger.info("插入示例训练指标...")
        for step in range(1, 6):
            metrics_logger.insert_metric(
                experiment_name="gsm8k_grpo_demo",
                step=step * 100,
                loss=1.5 - step * 0.2,
                reward=0.5 + step * 0.1,
            )

        logger.info("✓ 示例数据插入成功")

        # 4. 查询验证
        logger.info("查询最近 5 条记录...")
        with metrics_logger.connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT experiment_name, step, loss, reward, timestamp
                FROM training_metrics
                ORDER BY timestamp DESC
                LIMIT 5
                """
            )
            results = cursor.fetchall()
            for row in results:
                logger.info(
                    f"  {row['experiment_name']} | "
                    f"step={row['step']} | "
                    f"loss={row['loss']:.3f} | "
                    f"reward={row['reward']:.3f} | "
                    f"time={row['timestamp']}"
                )

        logger.info("=== 示例执行完成 ===")

    except Exception as e:
        logger.error(f"执行失败: {e}")
        raise
    finally:
        # 5. 关闭连接
        metrics_logger.close()


if __name__ == "__main__":
    main()
