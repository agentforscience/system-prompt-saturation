"""Reserve GPU memory on a shared host in 1 GB chunks.

The target size (GB) is read from a control file every 0.1 s; chunks are freed/re-grabbed to
match it, so a job of mine can shrink the reservation and take the memory immediately.
Usage: CUDA_VISIBLE_DEVICES=2 python src/hold_gpu.py logs/hold_gpu2.target
"""
import os, sys, time, torch
ctl = sys.argv[1]
chunks = []
while True:
    try:
        target = int(float(open(ctl).read().strip()))
    except Exception:
        target = len(chunks)
    if target < 0:
        break
    while len(chunks) > target:
        chunks.pop()
        torch.cuda.empty_cache()
    if len(chunks) < target:
        try:
            chunks.append(torch.empty(2**30, dtype=torch.uint8, device="cuda"))
            continue
        except torch.OutOfMemoryError:
            pass
    time.sleep(0.1)
