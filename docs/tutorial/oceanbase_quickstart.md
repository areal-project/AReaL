# OceanBase 集成快速入门

本指南介绍如何将 AReaL 训练指标持久化到 OceanBase 数据库。

## 什么是 OceanBase？

[OceanBase](https://www.oceanbase.com/) 是一个开源的分布式关系数据库，兼容 MySQL 协议，具有以下特点：

- **高性能**：支持百万级 TPS
- **高可用**：多副本强一致性
- **MySQL 兼容**：可使用 MySQL 客户端和工具
- **水平扩展**：支持分布式架构

适合存储大规模训练指标、实验记录和模型元数据。

---

## 准备工作

### 1. 安装 OceanBase

#### 方式 A：Docker 部署（推荐用于开发测试）

```bash
# 拉取 OceanBase 社区版镜像
docker pull oceanbase/oceanbase-ce:latest

# 启动容器
docker run -d \
  --name oceanbase-ce \
  -p 2881:2881 \
  -p 2882:2882 \
  -e MODE=mini \
  oceanbase/oceanbase-ce:latest

# 等待 2-3 分钟让服务完全启动
docker logs -f oceanbase-ce
```

#### 方式 B：生产环境部署

参考 [OceanBase 官方文档](https://www.oceanbase.com/docs/oceanbase-database)。

### 2. 安装 Python 依赖

```bash
cd /path/to/AReaL
uv pip install pymysql
```

或在 `pyproject.toml` 中添加依赖后运行 `uv sync`。

---

## 连接配置

### 环境变量配置（推荐）

```bash
export OB_HOST=127.0.0.1
export OB_PORT=2881
export OB_USER=root@test
export OB_PASSWORD=""
export OB_DATABASE=test
```

### 默认连接参数

| 参数       | 默认值       | 说明                                   |
| ---------- | ------------ | -------------------------------------- |
| `host`     | `127.0.0.1`  | OceanBase 主机地址                     |
| `port`     | `2881`       | MySQL 协议端口                         |
| `user`     | `root@test`  | 用户名（格式: `user@tenant`）          |
| `password` | `""`         | 密码（Docker 默认为空）                |
| `database` | `test`       | 数据库名                               |

---

## 运行示例

### 基础示例

```bash
cd /path/to/AReaL
python examples/utils/oceanbase_example.py
```

**输出示例**：

```
(AReaL) 20260131-14:30:00.123 OceanBaseExample INFO: === OceanBase 集成示例 ===
(AReaL) 20260131-14:30:00.124 OceanBaseExample INFO: 连接配置: 127.0.0.1:2881/test
(AReaL) 20260131-14:30:00.256 OceanBaseExample INFO: 成功连接到 OceanBase: 127.0.0.1:2881/test
(AReaL) 20260131-14:30:00.312 OceanBaseExample INFO: 成功创建表 training_metrics
(AReaL) 20260131-14:30:00.345 OceanBaseExample INFO: 插入指标: gsm8k_grpo_demo step=100 loss=1.3 reward=0.6
...
(AReaL) 20260131-14:30:00.512 OceanBaseExample INFO: === 示例执行完成 ===
```

---

## 集成到训练脚本

### 方式 1：直接集成

在训练脚本中导入并使用 `OceanBaseMetricsLogger`：

```python
from examples.utils.oceanbase_example import OceanBaseMetricsLogger

# 初始化
metrics_logger = OceanBaseMetricsLogger(
    host="127.0.0.1",
    port=2881,
    user="root@test",
    password="",
    database="test",
)

# 连接并创建表
metrics_logger.connect()
metrics_logger.create_table()

# 在训练循环中记录指标
for step in range(num_steps):
    # ... 训练代码 ...

    metrics_logger.insert_metric(
        experiment_name="my_experiment",
        step=step,
        loss=loss.item(),
        reward=reward.mean().item(),
    )

# 训练结束后关闭连接
metrics_logger.close()
```

### 方式 2：扩展为自定义 Logger

创建自定义日志类继承 `OceanBaseMetricsLogger`：

```python
from examples.utils.oceanbase_example import OceanBaseMetricsLogger

class CustomMetricsLogger(OceanBaseMetricsLogger):
    def log_training_step(self, experiment_name, step, metrics_dict):
        """记录训练步骤的所有指标"""
        self.insert_metric(
            experiment_name=experiment_name,
            step=step,
            loss=metrics_dict.get("loss"),
            reward=metrics_dict.get("reward"),
        )

    def log_evaluation(self, experiment_name, step, eval_metrics):
        """记录评估指标"""
        # 自定义评估指标记录逻辑
        pass
```

---

## 常见查询

### 查询实验的训练曲线

```sql
SELECT step, loss, reward, timestamp
FROM training_metrics
WHERE experiment_name = 'gsm8k_grpo_demo'
ORDER BY step;
```

### 查询最近的训练记录

```sql
SELECT *
FROM training_metrics
ORDER BY timestamp DESC
LIMIT 10;
```

### 计算平均损失

```sql
SELECT
    experiment_name,
    AVG(loss) as avg_loss,
    MIN(loss) as min_loss,
    MAX(loss) as max_loss
FROM training_metrics
GROUP BY experiment_name;
```

### 按时间范围查询

```sql
SELECT *
FROM training_metrics
WHERE timestamp >= '2026-01-31 00:00:00'
  AND timestamp < '2026-02-01 00:00:00';
```

---

## 故障排除

### 问题 1：连接失败

**错误信息**：
```
pymysql.err.OperationalError: (2003, "Can't connect to MySQL server on '127.0.0.1'")
```

**解决方案**：
1. 检查 OceanBase 服务是否启动：
   ```bash
   docker ps | grep oceanbase
   ```
2. 检查端口是否正确（默认 2881）
3. 检查防火墙设置

### 问题 2：认证失败

**错误信息**：
```
pymysql.err.OperationalError: (1045, "Access denied for user 'root@test'")
```

**解决方案**：
1. 确认用户名格式为 `user@tenant`（如 `root@test`）
2. 检查密码是否正确
3. Docker 部署默认密码为空

### 问题 3：表已存在

**错误信息**：
```
pymysql.err.InternalError: (1050, "Table 'training_metrics' already exists")
```

**解决方案**：
示例代码使用 `CREATE TABLE IF NOT EXISTS`，不会报错。如需重建表：

```sql
DROP TABLE IF EXISTS training_metrics;
```

### 问题 4：性能问题

**症状**：插入速度慢

**优化方案**：
1. **批量插入**：
   ```python
   def insert_metrics_batch(self, metrics_list):
       insert_sql = """
       INSERT INTO training_metrics
       (experiment_name, step, loss, reward, timestamp)
       VALUES (%s, %s, %s, %s, %s)
       """
       with self.connection.cursor() as cursor:
           cursor.executemany(insert_sql, metrics_list)
   ```

2. **异步写入**：使用队列缓冲指标，定期批量写入

3. **索引优化**：根据查询模式调整索引

---

## 高级配置

### 连接池

对于高并发场景，使用连接池：

```python
from pymysql.pooling import PooledConnection

# 创建连接池（需要额外配置）
# 参考 pymysql 文档
```

### 分区表

对于大规模数据，使用分区表：

```sql
CREATE TABLE training_metrics (
    id BIGINT AUTO_INCREMENT,
    experiment_name VARCHAR(100) NOT NULL,
    step INT NOT NULL,
    loss FLOAT,
    reward FLOAT,
    timestamp DATETIME NOT NULL,
    PRIMARY KEY (id, timestamp)
) PARTITION BY RANGE (TO_DAYS(timestamp)) (
    PARTITION p202601 VALUES LESS THAN (TO_DAYS('2026-02-01')),
    PARTITION p202602 VALUES LESS THAN (TO_DAYS('2026-03-01')),
    PARTITION p202603 VALUES LESS THAN (TO_DAYS('2026-04-01'))
);
```

---

## 相关资源

- [OceanBase 官方文档](https://www.oceanbase.com/docs/)
- [PyMySQL 文档](https://pymysql.readthedocs.io/)
- [AReaL 日志系统](../../areal/utils/logging.py)
- [示例代码](../../examples/utils/oceanbase_example.py)

---

## 下一步

- 集成到您的训练脚本
- 配置生产环境的 OceanBase 集群
- 使用 Grafana 可视化训练指标
- 探索 OceanBase 的高级特性（分布式事务、读写分离等）
