"""Greedy generation for every prompt in data/prompts.jsonl with one model (vLLM, in-process engine).

Usage: CUDA_VISIBLE_DEVICES=2 python src/generate.py --model Qwen/Qwen2.5-7B-Instruct --name qwen2.5-7b --gpu-gb 20 \
           [--holder-ctl logs/hold_gpu2.target --holder-gb 40]

Outputs append to results/gen/<name>.jsonl (resumable: finished uids are skipped).
Shared-GPU race fix: memory is grabbed with a placeholder tensor as soon as it is free (optionally after shrinking
my own holder process), and released inside vLLM right before it measures free memory.
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HOME", "/data/transformers")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNK = 900
MAX_TOKENS = 1024


def gpu_total_gb():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.check_output(["nvidia-smi", "-i", dev, "--query-gpu=memory.total",
                                   "--format=csv,noheader,nounits"]).decode()
    return float(out.strip()) / 1024


def gpu_free_gb():
    dev = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    out = subprocess.check_output(["nvidia-smi", "-i", dev, "--query-gpu=memory.free",
                                   "--format=csv,noheader,nounits"]).decode()
    return float(out.strip()) / 1024


def holder_held_gb(ctl):
    """GB actually held by my holder process (its target file may be larger than what it managed to grab)."""
    try:
        pid = subprocess.check_output(["pgrep", "-f", "hold_gpu.py " + ctl]).decode().split()[0]
        for line in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                             "--format=csv,noheader,nounits"]).decode().splitlines():
            p, mem = [x.strip() for x in line.split(",")]
            if p == pid:
                return max(0, int(float(mem) / 1024))
    except Exception:
        pass
    return int(float(open(ctl).read().strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--gpu-gb", type=float, default=4.5)
    ap.add_argument("--max-model-len", type=int, default=10752)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--batched-tokens", type=int, default=4096)
    ap.add_argument("--exps", default="A,B,D,E,F")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: items per cell")
    ap.add_argument("--holder-ctl", default="", help="control file of my holder on this GPU")
    ap.add_argument("--holder-gb", type=int, default=40, help="holder size to restore when done")
    ap.add_argument("--eager", type=int, default=1)
    ap.add_argument("--kv-gb", type=float, default=0, help="fixed KV-cache size (skips memory profiling, which "
                    "mis-attributes other tenants' memory growth to this process on shared GPUs)")
    args = ap.parse_args()

    exps = set(args.exps.split(","))
    prompts = [json.loads(l) for l in open(os.path.join(ROOT, "data", "prompts.jsonl"))]
    prompts = [p for p in prompts if p["exp"] in exps and (not args.limit or p["item"] < args.limit)]
    out_dir = os.path.join(ROOT, "results", "gen")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.name}{'_smoke' if args.limit else ''}.jsonl")
    # one writer per output file: a second queue that reaches the same model moves on
    import fcntl
    lock_f = open(out_path + ".lock", "w")
    try:
        fcntl.flock(lock_f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"[{args.name}] LOCKED by another process; 0 to do here", flush=True)
        return 0
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)["uid"] for l in open(out_path)}
    todo = [p for p in prompts if p["uid"] not in done]
    print(f"[{args.name}] {len(todo)} to do ({len(done)} done)", flush=True)
    if not todo:
        return 0

    import torch
    import vllm.v1.worker.gpu_worker as gw
    try:
        import vllm.utils.mem_utils as mu
    except ImportError:
        mu = None
    from vllm import LLM, SamplingParams

    util = min(0.95, args.gpu_gb / gpu_total_gb())
    if args.holder_ctl:
        held = holder_held_gb(args.holder_ctl)
        need = int(args.gpu_gb + 1.5) + 1
        open(args.holder_ctl, "w").write(str(max(0, held - need)))
        # wait for the holder to free its chunks before creating a CUDA context (context creation fails hard on a
        # full GPU)
        t_wait = time.time()
        while gpu_free_gb() < min(need, held) - 1 and time.time() - t_wait < 15:
            time.sleep(0.1)
    hold, deadline = [], time.time() + 600
    while not hold and time.time() < deadline:
        try:
            hold.append(torch.empty(int((args.gpu_gb + 0.3) * 1024 ** 3), dtype=torch.uint8, device="cuda"))
        except (torch.OutOfMemoryError, RuntimeError):
            time.sleep(0.2)
    if not hold:
        print("HOLD_TIMEOUT: could not reserve GPU memory", flush=True)
        return 75
    _snapshot = gw.MemorySnapshot

    class _ReleaseThenSnapshot(_snapshot):
        def __init__(self, *a, **k):
            if hold:
                hold.clear()
                torch.cuda.empty_cache()
            super().__init__(*a, **k)
    gw.MemorySnapshot = _ReleaseThenSnapshot
    if mu is not None and hasattr(mu, "MemorySnapshot"):
        mu.MemorySnapshot = _ReleaseThenSnapshot

    llm = LLM(model=args.model, gpu_memory_utilization=util, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.batched_tokens,
              enforce_eager=bool(args.eager), enable_prefix_caching=True, seed=42, trust_remote_code=False,
              **({"kv_cache_memory_bytes": int(args.kv_gb * 1024 ** 3)} if args.kv_gb else {}))
    tok = llm.get_tokenizer()
    is_qwen3 = "qwen3" in args.model.lower()

    t0 = time.time()
    with open(out_path, "a") as fout:
        for s in range(0, len(todo), CHUNK):
            batch = todo[s:s + CHUNK]
            convs = [[{"role": "system", "content": p["system"]}, {"role": "user", "content": p["query"]}] for p in batch]
            sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, seed=42)
            kw = dict(chat_template_kwargs={"enable_thinking": False}) if is_qwen3 else {}
            outs = llm.chat(convs, sp, use_tqdm=False, **kw)
            for p, o in zip(batch, outs):
                c = o.outputs[0]
                fout.write(json.dumps(dict(uid=p["uid"], text=c.text, n_tokens=len(c.token_ids),
                                           n_prompt_tokens=len(o.prompt_token_ids), finish=c.finish_reason),
                                      ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{args.name}] {s + len(batch)}/{len(todo)} done, {time.time() - t0:.0f}s", flush=True)
    with open(os.path.join(out_dir, f"{args.name}_template_example.txt"), "w") as f:
        f.write(tok.apply_chat_template([{"role": "system", "content": todo[0]["system"]},
                                         {"role": "user", "content": todo[0]["query"]}],
                                        tokenize=False, add_generation_prompt=True,
                                        **({"enable_thinking": False} if is_qwen3 else {})))
    print(f"[{args.name}] FINISHED in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code or 0)
