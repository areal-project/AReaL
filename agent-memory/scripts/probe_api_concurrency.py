"""Probe matrixllm gateway concurrency for gpt-5-mini.

Fires N concurrent chat requests, measures per-request latency, throughput,
and counts rate-limit / error responses. Read-only: no side effects.
"""
import argparse
import concurrent.futures as cf
import time

from openai import OpenAI

API_KEY = "sk-43dd5f664179406d92fec42a9364f8a5"
BASE_URL = "https://matrixllm.alipay.com/v1/"


def one_request(client, model, idx, max_tokens):
    prompt = (
        "You are in a simulated household. Your task is to put a clean mug in the "
        "coffee machine. List the first 3 actions you would take. Be concise."
    )
    t0 = time.time()
    is_gpt5 = model.startswith("gpt-5")
    kwargs = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    if is_gpt5:
        kwargs["max_completion_tokens"] = max_tokens  # gpt-5 rejects max_tokens
        # gpt-5 only supports default temperature (1.0); omit temperature entirely
    else:
        kwargs["max_tokens"] = max_tokens
        kwargs["temperature"] = 0
    try:
        r = client.chat.completions.create(**kwargs)
        dt = time.time() - t0
        ntok = r.usage.completion_tokens if r.usage else 0
        return {"idx": idx, "ok": True, "dt": dt, "tok": ntok, "err": None}
    except Exception as e:
        dt = time.time() - t0
        return {"idx": idx, "ok": False, "dt": dt, "tok": 0, "err": f"{type(e).__name__}: {str(e)[:150]}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5-mini")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    print(f"Probing model={args.model} concurrency={args.concurrency} max_tokens={args.max_tokens}")
    wall0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_request, client, args.model, i, args.max_tokens)
                for i in range(args.concurrency)]
        results = [f.result() for f in cf.as_completed(futs)]
    wall = time.time() - wall0

    ok = [r for r in results if r["ok"]]
    bad = [r for r in results if not r["ok"]]
    lat = sorted(r["dt"] for r in ok)
    tot_tok = sum(r["tok"] for r in ok)

    print(f"\n=== Results ({len(ok)}/{len(results)} ok) ===")
    print(f"wall time:        {wall:.1f}s")
    if ok:
        print(f"latency min/med/max: {lat[0]:.1f} / {lat[len(lat)//2]:.1f} / {lat[-1]:.1f}s")
        print(f"throughput:       {tot_tok/wall:.0f} completion tok/s aggregate")
        print(f"req throughput:   {len(ok)/wall:.1f} req/s")
    if bad:
        print(f"\n=== {len(bad)} FAILURES (rate limit / error) ===")
        from collections import Counter
        c = Counter(r["err"].split(":")[0] for r in bad)
        for k, v in c.most_common():
            print(f"  {v}x  {k}")
        print("  sample:", bad[0]["err"])


if __name__ == "__main__":
    main()
