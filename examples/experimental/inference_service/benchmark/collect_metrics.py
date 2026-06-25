#!/usr/bin/env python3
"""Snapshot and diff SGLang Prometheus metrics.

Usage:
    # Snapshot current metrics
    python collect_sglang_metrics.py snapshot http://127.0.0.1:30000

    # Diff two snapshot files → JSON with deltas + derived throughput
    python collect_sglang_metrics.py diff pre.json post.json --wall-clock 120
"""

import argparse
import json
import re
import sys
import urllib.request

COUNTER_KEYS = [
    "sglang:prompt_tokens_total",
    "sglang:generation_tokens_total",
    "sglang:num_requests_total",
    "sglang:e2e_request_latency_seconds_sum",
    "sglang:e2e_request_latency_seconds_count",
    "sglang:time_to_first_token_seconds_sum",
    "sglang:time_to_first_token_seconds_count",
    "sglang:queue_time_seconds_sum",
    "sglang:queue_time_seconds_count",
]


def fetch_metrics(base_url: str) -> dict:
    url = base_url.rstrip("/")
    if "/v1" in url:
        url = url.split("/v1")[0]
    url = url + "/metrics"

    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        text = resp.read().decode()

    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"([\w:]+)(?:\{([^}]*)\})?\s+([\d.eE+\-]+)", line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        key = f"{name}{{{labels}}}" if labels else name
        metrics[key] = value

    return metrics


def extract_scalar(metrics: dict, prefix: str) -> float:
    for key, val in metrics.items():
        if key.startswith(prefix) and "{" not in key:
            return val
        if key.startswith(prefix + "{"):
            return val
    return 0.0


def snapshot(base_url: str) -> dict:
    raw = fetch_metrics(base_url)
    result = {}
    for k in COUNTER_KEYS:
        result[k] = extract_scalar(raw, k)
    return result


def diff(pre: dict, post: dict, wall_clock: float) -> dict:
    delta = {}
    for k in COUNTER_KEYS:
        delta[k] = post.get(k, 0) - pre.get(k, 0)

    prompt_tokens = delta.get("sglang:prompt_tokens_total", 0)
    gen_tokens = delta.get("sglang:generation_tokens_total", 0)
    total_tokens = prompt_tokens + gen_tokens
    n_requests = delta.get("sglang:num_requests_total", 0)
    e2e_sum = delta.get("sglang:e2e_request_latency_seconds_sum", 0)
    e2e_count = delta.get("sglang:e2e_request_latency_seconds_count", 0)
    ttft_sum = delta.get("sglang:time_to_first_token_seconds_sum", 0)
    ttft_count = delta.get("sglang:time_to_first_token_seconds_count", 0)
    queue_sum = delta.get("sglang:queue_time_seconds_sum", 0)
    queue_count = delta.get("sglang:queue_time_seconds_count", 0)

    return {
        "prompt_tokens": int(prompt_tokens),
        "generation_tokens": int(gen_tokens),
        "total_tokens": int(total_tokens),
        "num_requests": int(n_requests),
        "wall_clock_seconds": round(wall_clock, 1),
        "input_throughput_tok_per_sec": round(prompt_tokens / wall_clock, 1)
        if wall_clock > 0
        else 0,
        "output_throughput_tok_per_sec": round(gen_tokens / wall_clock, 1)
        if wall_clock > 0
        else 0,
        "total_throughput_tok_per_sec": round(total_tokens / wall_clock, 1)
        if wall_clock > 0
        else 0,
        "request_throughput_per_sec": round(n_requests / wall_clock, 2)
        if wall_clock > 0
        else 0,
        "avg_prompt_tokens_per_req": round(prompt_tokens / n_requests, 1)
        if n_requests > 0
        else 0,
        "avg_gen_tokens_per_req": round(gen_tokens / n_requests, 1)
        if n_requests > 0
        else 0,
        "avg_e2e_latency_seconds": round(e2e_sum / e2e_count, 2)
        if e2e_count > 0
        else 0,
        "avg_ttft_seconds": round(ttft_sum / ttft_count, 3) if ttft_count > 0 else 0,
        "avg_queue_time_seconds": round(queue_sum / queue_count, 3)
        if queue_count > 0
        else 0,
        "total_llm_time_seconds": round(e2e_sum, 1),
    }


def monitor(base_url: str, interval: float, output_file: str):
    """Poll SGLang /metrics every `interval` seconds, write CSV for peak throughput analysis."""
    import csv
    import signal
    import time

    fieldnames = [
        "timestamp",
        "elapsed",
        "prompt_tokens",
        "generation_tokens",
        "num_requests",
        "running_reqs",
        "queue_reqs",
        "input_tok_per_sec",
        "output_tok_per_sec",
    ]

    stop = [False]

    def _handle(sig, frame):
        stop[0] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    prev = None
    prev_time = None
    t0 = time.time()

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while not stop[0]:
            try:
                raw = fetch_metrics(base_url)
                now = time.time()
                cur_prompt = extract_scalar(raw, "sglang:prompt_tokens_total")
                cur_gen = extract_scalar(raw, "sglang:generation_tokens_total")
                cur_reqs = extract_scalar(raw, "sglang:num_requests_total")
                running = extract_scalar(raw, "sglang:num_running_reqs")
                queue = extract_scalar(raw, "sglang:num_queue_reqs")

                input_rate = 0.0
                output_rate = 0.0
                if prev is not None and prev_time is not None:
                    dt = now - prev_time
                    if dt > 0:
                        input_rate = (cur_prompt - prev["prompt_tokens"]) / dt
                        output_rate = (cur_gen - prev["generation_tokens"]) / dt

                row = {
                    "timestamp": round(now, 2),
                    "elapsed": round(now - t0, 1),
                    "prompt_tokens": int(cur_prompt),
                    "generation_tokens": int(cur_gen),
                    "num_requests": int(cur_reqs),
                    "running_reqs": int(running),
                    "queue_reqs": int(queue),
                    "input_tok_per_sec": round(input_rate, 1),
                    "output_tok_per_sec": round(output_rate, 1),
                }
                writer.writerow(row)
                f.flush()

                prev = {"prompt_tokens": cur_prompt, "generation_tokens": cur_gen}
                prev_time = now
            except Exception:
                pass

            time.sleep(interval)


def percentile(sorted_vals: list, p: float) -> float:
    """Compute p-th percentile from a sorted list (0 <= p <= 100)."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def analyze_monitor_csv(csv_path: str) -> dict:
    """Analyze monitor CSV to extract throughput stats and concurrency distribution."""
    import csv as csvmod

    rows = []
    with open(csv_path) as f:
        for r in csvmod.DictReader(f):
            rows.append({k: float(v) for k, v in r.items()})

    if len(rows) < 2:
        return {}

    input_rates = [r["input_tok_per_sec"] for r in rows if r["input_tok_per_sec"] > 0]
    output_rates = [
        r["output_tok_per_sec"] for r in rows if r["output_tok_per_sec"] > 0
    ]
    running = [r["running_reqs"] for r in rows]
    queued = [r["queue_reqs"] for r in rows]

    total_prompt = rows[-1]["prompt_tokens"] - rows[0]["prompt_tokens"]
    total_gen = rows[-1]["generation_tokens"] - rows[0]["generation_tokens"]
    total_reqs = rows[-1]["num_requests"] - rows[0]["num_requests"]
    total_time = rows[-1]["elapsed"] - rows[0]["elapsed"]

    sorted_running = sorted(running)
    sorted_queued = sorted(queued)
    sorted_input = sorted(input_rates) if input_rates else []

    return {
        "monitor_duration_seconds": round(total_time, 1),
        "monitor_samples": len(rows),
        "total_prompt_tokens": int(total_prompt),
        "total_generation_tokens": int(total_gen),
        "total_requests": int(total_reqs),
        # Throughput
        "avg_input_throughput_tok_per_sec": round(total_prompt / total_time, 1)
        if total_time > 0
        else 0,
        "avg_output_throughput_tok_per_sec": round(total_gen / total_time, 1)
        if total_time > 0
        else 0,
        "peak_input_throughput_tok_per_sec": round(max(input_rates), 1)
        if input_rates
        else 0,
        "peak_output_throughput_tok_per_sec": round(max(output_rates), 1)
        if output_rates
        else 0,
        "p50_input_throughput_tok_per_sec": round(percentile(sorted_input, 50), 1)
        if sorted_input
        else 0,
        "p95_input_throughput_tok_per_sec": round(percentile(sorted_input, 95), 1)
        if sorted_input
        else 0,
        "p99_input_throughput_tok_per_sec": round(percentile(sorted_input, 99), 1)
        if sorted_input
        else 0,
        # Concurrency distribution (running requests at SGLang)
        "avg_running_reqs": round(sum(running) / len(running), 1) if running else 0,
        "max_running_reqs": int(max(running)) if running else 0,
        "min_running_reqs": int(min(running)) if running else 0,
        "p50_running_reqs": round(percentile(sorted_running, 50), 1),
        "p75_running_reqs": round(percentile(sorted_running, 75), 1),
        "p95_running_reqs": round(percentile(sorted_running, 95), 1),
        "p99_running_reqs": round(percentile(sorted_running, 99), 1),
        # Queue distribution
        "avg_queue_reqs": round(sum(queued) / len(queued), 1) if queued else 0,
        "max_queue_reqs": int(max(queued)) if queued else 0,
        "p50_queue_reqs": round(percentile(sorted_queued, 50), 1),
        "p95_queue_reqs": round(percentile(sorted_queued, 95), 1),
        "p99_queue_reqs": round(percentile(sorted_queued, 99), 1),
    }


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    snap_p = sub.add_parser("snapshot")
    snap_p.add_argument("url", help="SGLang base URL (e.g. http://127.0.0.1:30000)")
    snap_p.add_argument("-o", "--output", help="Output file (default: stdout)")

    diff_p = sub.add_parser("diff")
    diff_p.add_argument("pre", help="Pre-snapshot JSON")
    diff_p.add_argument("post", help="Post-snapshot JSON")
    diff_p.add_argument(
        "--wall-clock", type=float, required=True, help="Wall clock seconds"
    )
    diff_p.add_argument("-o", "--output", help="Output file (default: stdout)")

    mon_p = sub.add_parser("monitor")
    mon_p.add_argument("url", help="SGLang base URL")
    mon_p.add_argument(
        "-i", "--interval", type=float, default=5.0, help="Poll interval (default: 5s)"
    )
    mon_p.add_argument("-o", "--output", required=True, help="Output CSV file")

    ana_p = sub.add_parser("analyze")
    ana_p.add_argument("csv", help="Monitor CSV file")
    ana_p.add_argument("-o", "--output", help="Output JSON file (default: stdout)")

    args = parser.parse_args()

    if args.cmd == "snapshot":
        result = snapshot(args.url)
        out = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
        else:
            print(out)

    elif args.cmd == "diff":
        with open(args.pre) as f:
            pre = json.load(f)
        with open(args.post) as f:
            post = json.load(f)
        result = diff(pre, post, args.wall_clock)
        out = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
        else:
            print(out)

    elif args.cmd == "monitor":
        monitor(args.url, args.interval, args.output)

    elif args.cmd == "analyze":
        result = analyze_monitor_csv(args.csv)
        out = json.dumps(result, indent=2)
        if args.output:
            with open(args.output, "w") as f:
                f.write(out)
        else:
            print(out)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
