#!/usr/bin/env python3
"""
跨Benchmark实验结果分析脚本

功能:
1. 收集并汇总各benchmark的实验结果
2. 计算memory迁移前后的性能对比
3. 生成可视化报告
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import glob


def find_experiment_dirs(results_dir: str, experiment_name: str) -> Dict[str, List[Path]]:
    """查找所有相关的实验目录"""
    results_path = Path(results_dir)
    experiments = {}

    for benchmark_dir in results_path.iterdir():
        if not benchmark_dir.is_dir():
            continue

        benchmark_name = benchmark_dir.name
        exp_dirs = []

        # 查找匹配的实验目录
        for exp_dir in benchmark_dir.glob(f"exp_{experiment_name}*"):
            if exp_dir.is_dir():
                exp_dirs.append(exp_dir)

        # 也检查bigcodebench_eval等特殊目录结构
        for special_dir in benchmark_dir.glob("**/memory/*"):
            if special_dir.is_dir() and experiment_name in str(special_dir):
                exp_dirs.append(special_dir)

        if exp_dirs:
            experiments[benchmark_name] = sorted(exp_dirs, key=lambda x: x.stat().st_mtime, reverse=True)

    return experiments


def load_metrics_from_dir(exp_dir: Path) -> Optional[Dict[str, Any]]:
    """从实验目录加载指标"""
    metrics = {
        "path": str(exp_dir),
        "timestamp": None,
        "success_rate": None,
        "total_tasks": None,
        "successful_tasks": None,
        "memory_count": None,
        "avg_q_value": None,
        "epochs": [],
    }

    # 尝试多种可能的结果文件格式
    possible_files = [
        "results.json",
        "metrics.json",
        "summary.json",
        "experiment_results.json",
        "eval_results.json",
    ]

    for filename in possible_files:
        result_file = exp_dir / filename
        if result_file.exists():
            try:
                with open(result_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    metrics.update(_extract_metrics(data))
                    break
            except (json.JSONDecodeError, Exception) as e:
                print(f"[WARN] 无法解析 {result_file}: {e}")

    # 查找epoch级别的结果
    epoch_dirs = list(exp_dir.glob("epoch_*")) + list(exp_dir.glob("section_*"))
    for epoch_dir in sorted(epoch_dirs):
        epoch_metrics = load_metrics_from_dir(epoch_dir)
        if epoch_metrics and epoch_metrics.get("success_rate") is not None:
            epoch_num = epoch_dir.name.split("_")[-1]
            metrics["epochs"].append({
                "epoch": epoch_num,
                **epoch_metrics
            })

    # 从local_cache或其他位置读取额外信息
    cache_dir = exp_dir / "local_cache"
    if cache_dir.exists():
        try:
            q_cache_file = cache_dir / "q_cache.json"
            if q_cache_file.exists():
                with open(q_cache_file, "r", encoding="utf-8") as f:
                    q_data = json.load(f)
                    if q_data:
                        q_values = [v.get("q_value", 0) for v in q_data.values() if isinstance(v, dict)]
                        if q_values:
                            metrics["avg_q_value"] = sum(q_values) / len(q_values)
                            metrics["memory_count"] = len(q_values)
        except Exception:
            pass

    return metrics


def _extract_metrics(data: Dict) -> Dict[str, Any]:
    """从结果数据中提取关键指标"""
    extracted = {}

    # 通用字段映射
    field_mappings = {
        "success_rate": ["success_rate", "accuracy", "pass_rate", "score", "success"],
        "total_tasks": ["total", "total_tasks", "num_tasks", "n_samples"],
        "successful_tasks": ["successful", "passed", "correct", "success_count"],
    }

    for target, sources in field_mappings.items():
        for source in sources:
            if source in data:
                extracted[target] = data[source]
                break

    # 处理嵌套结构
    if "results" in data:
        nested = _extract_metrics(data["results"])
        for k, v in nested.items():
            if k not in extracted or extracted[k] is None:
                extracted[k] = v

    if "metrics" in data:
        nested = _extract_metrics(data["metrics"])
        for k, v in nested.items():
            if k not in extracted or extracted[k] is None:
                extracted[k] = v

    return extracted


def calculate_transfer_gain(
    source_metrics: Dict[str, Any],
    target_metrics: Dict[str, Any],
    baseline_metrics: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """计算memory迁移带来的性能增益"""
    gain = {
        "source_success_rate": source_metrics.get("success_rate"),
        "target_success_rate": target_metrics.get("success_rate"),
        "absolute_gain": None,
        "relative_gain": None,
    }

    if baseline_metrics and baseline_metrics.get("success_rate") is not None:
        gain["baseline_success_rate"] = baseline_metrics.get("success_rate")

    src_rate = source_metrics.get("success_rate")
    tgt_rate = target_metrics.get("success_rate")

    if src_rate is not None and tgt_rate is not None:
        gain["absolute_gain"] = tgt_rate - (baseline_metrics.get("success_rate", 0) if baseline_metrics else 0)
        if baseline_metrics and baseline_metrics.get("success_rate"):
            baseline = baseline_metrics["success_rate"]
            if baseline > 0:
                gain["relative_gain"] = (tgt_rate - baseline) / baseline * 100

    return gain


def generate_report(
    results_dir: str,
    experiment_name: str,
    source_benchmark: str,
    target_benchmarks: List[str],
) -> Dict[str, Any]:
    """生成完整的实验报告"""
    report = {
        "experiment_name": experiment_name,
        "generated_at": datetime.now().isoformat(),
        "source_benchmark": source_benchmark,
        "target_benchmarks": target_benchmarks,
        "source_results": None,
        "target_results": {},
        "transfer_analysis": {},
        "summary": {},
    }

    # 查找实验目录
    exp_dirs = find_experiment_dirs(results_dir, experiment_name)
    print(f"[INFO] 找到的实验目录: {list(exp_dirs.keys())}")

    # 加载源benchmark结果
    if source_benchmark in exp_dirs and exp_dirs[source_benchmark]:
        source_dir = exp_dirs[source_benchmark][0]
        report["source_results"] = load_metrics_from_dir(source_dir)
        print(f"[INFO] 源benchmark ({source_benchmark}) 结果: {report['source_results']}")

    # 加载目标benchmark结果
    for target in target_benchmarks:
        if target in exp_dirs and exp_dirs[target]:
            target_dir = exp_dirs[target][0]
            target_metrics = load_metrics_from_dir(target_dir)
            report["target_results"][target] = target_metrics

            # 计算迁移增益
            if report["source_results"]:
                report["transfer_analysis"][target] = calculate_transfer_gain(
                    report["source_results"],
                    target_metrics
                )

            print(f"[INFO] 目标benchmark ({target}) 结果: {target_metrics}")

    # 生成摘要
    report["summary"] = generate_summary(report)

    return report


def generate_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    """生成实验摘要"""
    summary = {
        "total_benchmarks_evaluated": len(report["target_results"]),
        "successful_transfers": 0,
        "avg_success_rate_improvement": None,
        "best_transfer": None,
        "worst_transfer": None,
    }

    improvements = []
    for target, analysis in report["transfer_analysis"].items():
        if analysis.get("absolute_gain") is not None:
            improvements.append({
                "benchmark": target,
                "gain": analysis["absolute_gain"]
            })
            if analysis["absolute_gain"] > 0:
                summary["successful_transfers"] += 1

    if improvements:
        avg_gain = sum(i["gain"] for i in improvements) / len(improvements)
        summary["avg_success_rate_improvement"] = avg_gain
        summary["best_transfer"] = max(improvements, key=lambda x: x["gain"])
        summary["worst_transfer"] = min(improvements, key=lambda x: x["gain"])

    return summary


def print_report(report: Dict[str, Any]) -> None:
    """打印格式化的报告"""
    print("\n" + "=" * 60)
    print("跨Benchmark Memory迁移实验报告")
    print("=" * 60)
    print(f"实验名称: {report['experiment_name']}")
    print(f"生成时间: {report['generated_at']}")
    print(f"源Benchmark: {report['source_benchmark']}")
    print(f"目标Benchmarks: {', '.join(report['target_benchmarks'])}")

    print("\n" + "-" * 60)
    print("源Benchmark训练结果")
    print("-" * 60)
    if report["source_results"]:
        src = report["source_results"]
        print(f"  成功率: {src.get('success_rate', 'N/A')}")
        print(f"  任务总数: {src.get('total_tasks', 'N/A')}")
        print(f"  Memory数量: {src.get('memory_count', 'N/A')}")
        print(f"  平均Q值: {src.get('avg_q_value', 'N/A')}")
    else:
        print("  未找到结果数据")

    print("\n" + "-" * 60)
    print("目标Benchmark评估结果")
    print("-" * 60)
    for target, metrics in report["target_results"].items():
        print(f"\n  [{target}]")
        if metrics:
            print(f"    成功率: {metrics.get('success_rate', 'N/A')}")
            print(f"    任务总数: {metrics.get('total_tasks', 'N/A')}")

            if target in report["transfer_analysis"]:
                analysis = report["transfer_analysis"][target]
                gain = analysis.get("absolute_gain")
                if gain is not None:
                    print(f"    绝对增益: {gain:+.2%}" if isinstance(gain, float) else f"    绝对增益: {gain}")
        else:
            print("    未找到结果数据")

    print("\n" + "-" * 60)
    print("实验摘要")
    print("-" * 60)
    summary = report["summary"]
    print(f"  评估的benchmark数量: {summary['total_benchmarks_evaluated']}")
    print(f"  成功迁移数量: {summary['successful_transfers']}")

    if summary.get("avg_success_rate_improvement") is not None:
        avg = summary["avg_success_rate_improvement"]
        print(f"  平均成功率提升: {avg:+.2%}" if isinstance(avg, float) else f"  平均成功率提升: {avg}")

    if summary.get("best_transfer"):
        best = summary["best_transfer"]
        print(f"  最佳迁移: {best['benchmark']} ({best['gain']:+.2%})" if isinstance(best['gain'], float) else f"  最佳迁移: {best['benchmark']} ({best['gain']})")

    if summary.get("worst_transfer"):
        worst = summary["worst_transfer"]
        print(f"  最差迁移: {worst['benchmark']} ({worst['gain']:+.2%})" if isinstance(worst['gain'], float) else f"  最差迁移: {worst['benchmark']} ({worst['gain']})")

    print("\n" + "=" * 60)


def save_report(report: Dict[str, Any], output_path: str) -> None:
    """保存报告到JSON文件"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"[INFO] 报告已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="分析跨Benchmark Memory迁移实验结果")
    parser.add_argument("--results_dir", required=True, help="结果目录")
    parser.add_argument("--experiment_name", required=True, help="实验名称")
    parser.add_argument("--source_benchmark", required=True, help="源benchmark")
    parser.add_argument("--target_benchmarks", nargs="+", required=True, help="目标benchmarks")
    parser.add_argument("--output", default=None, help="输出报告路径 (可选)")

    args = parser.parse_args()

    report = generate_report(
        results_dir=args.results_dir,
        experiment_name=args.experiment_name,
        source_benchmark=args.source_benchmark,
        target_benchmarks=args.target_benchmarks,
    )

    print_report(report)

    # 保存报告
    if args.output:
        save_report(report, args.output)
    else:
        # 默认保存位置
        default_output = Path(args.results_dir) / f"cross_benchmark_report_{args.experiment_name}.json"
        save_report(report, str(default_output))


if __name__ == "__main__":
    main()
