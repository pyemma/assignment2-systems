import pandas as pd

import torch

import triton
import triton.testing

from cs336_systems.flash_attention_2 import FlashAttentionPyTorch, FlashAttentionTriton


def fwd(cls, q, k, v):
    _ = cls.apply(q, k, v, True)
    return

def bwd(do, o):
    o.backward(do, retain_graph=True)
    return

def fwd_bwd(cls, q, k, v, do):
    o = cls.apply(q, k, v, True)
    o.backward(do, retain_graph=True)
    return

def benchmark_one(seq_len, d, precision, mode="fwd", class_name="triton"):
    q = torch.randn(1, seq_len, d, device="cuda", dtype=precision, requires_grad=True)
    k = torch.randn(1, seq_len, d, device="cuda", dtype=precision, requires_grad=True)
    v = torch.randn(1, seq_len, d, device="cuda", dtype=precision, requires_grad=True)
    do = torch.randn(1, seq_len, d, device="cuda", dtype=precision)
    cls = FlashAttentionTriton if class_name == "triton" else FlashAttentionPyTorch

    if mode == "fwd":
        ms = triton.testing.do_bench(lambda: fwd(cls, q, k, v), warmup=25, rep=100)
    elif mode == "bwd":
        o = cls.apply(q, k, v, True)
        ms = triton.testing.do_bench(lambda: bwd(do, o), warmup=25, rep=100, grad_to_none=[q, k, v])
    elif mode =="fwd_bwd":
        ms = triton.testing.do_bench(lambda: fwd_bwd(cls, q, k, v, do), warmup=25, rep=100, grad_to_none=[q, k, v])
    
    return ms


def benchmark(mode, class_name):
    result = []
    for seq_len in [128, 256, 512, 1024, 2048, 4096]:
        for d in [16, 32, 64, 128]:
            for precision in [torch.bfloat16, torch.float32]:
                ms = benchmark_one(seq_len, d, precision, mode, class_name)
                result.append((seq_len, d, "bfloat16" if precision == torch.bfloat16 else "float32", mode, class_name, ms))
    
    return pd.DataFrame(result, columns=["seq_len", "d", "precision", "mode", "class_name", "latency"])
