"""Measure loading and running Phi-2 on this computer's CPU.

The original app loaded Phi-2 in float32 on a CPU as soon as it started. Run one
dtype per process, so the memory numbers don't mix:

    python benchmarks/bench_phi_load.py --dtype float32
    python benchmarks/bench_phi_load.py --dtype bfloat16

Prints one JSON line: load time, peak memory, and generation speed.
"""
import argparse
import json
import platform
import time

import psutil
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "microsoft/phi-2"
PROMPT = "Instruct: Name three signs that several insurance claims come from one fraud ring.\nOutput:"


def rss_mb():
    return round(psutil.Process().memory_info().rss / 2**20)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()

    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=getattr(torch, args.dtype))
    load_seconds = time.perf_counter() - start
    rss_after_load = rss_mb()

    inputs = tokenizer(PROMPT, return_tensors="pt")
    start = time.perf_counter()
    output = model.generate(**inputs, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens,
                            do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generate_seconds = time.perf_counter() - start
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]

    result = {
        "model": MODEL,
        "dtype": args.dtype,
        "load_s": round(load_seconds, 2),
        "rss_mb_after_load": rss_after_load,
        "prompt_tokens": inputs["input_ids"].shape[1],
        "new_tokens": new_tokens,
        "generate_s": round(generate_seconds, 2),
        "tokens_per_s": round(new_tokens / generate_seconds, 2),
        "rss_mb_after_generate": rss_mb(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cpu": platform.processor(),
        "threads": torch.get_num_threads(),
    }
    memory = psutil.Process().memory_info()
    if hasattr(memory, "peak_wset"):  # Windows only: the most memory this process ever held
        result["peak_mb"] = round(memory.peak_wset / 2**20)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
