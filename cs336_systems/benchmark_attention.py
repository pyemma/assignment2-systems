import math
import timeit

import torch
import torch.cuda.nvtx as nvtx

from cs336_systems.annotated_sdpa import annotated_scaled_dot_product_attention

compiled_sdpa = torch.compile(annotated_scaled_dot_product_attention)

BATCH_SIZE = 8

@nvtx.range("attention forward")
def attention_forward(Q, K, V, mask=None, compiled=False):
    sdpa = compiled_sdpa if compiled else annotated_scaled_dot_product_attention
    output = sdpa(Q, K, V, mask)
    torch.cuda.synchronize()
    return output


@nvtx.range("attention backward")
def attention_backward(Q, K, V, mask=None, compiled=False):
    sdpa = compiled_sdpa if compiled else annotated_scaled_dot_product_attention
    output = sdpa(Q, K, V, mask)
    loss = output.sum()

    torch.cuda.reset_peak_memory_stats()
    bytes_used = torch.cuda.max_memory_allocated()
    gib_before = bytes_used / (1024 ** 3)

    start = timeit.default_timer()
    loss.backward()
    torch.cuda.synchronize()
    end = timeit.default_timer()

    bytes_used = torch.cuda.max_memory_allocated()
    gib_after = bytes_used / (1024 ** 3)
    mem = gib_after - gib_before

    return end - start, mem


def benchmark_forward(batch_size, seq_length, d_model, warmup_iters=5, benchmark_iters=100, compiled=False):
    Q = torch.randn(batch_size, seq_length, d_model, device="cuda")
    K = torch.randn(batch_size, seq_length, d_model, device="cuda")
    V = torch.randn(batch_size, seq_length, d_model, device="cuda")
    
    for _ in range(warmup_iters):
        attention_forward(Q, K ,V, compiled=compiled)
    
    times = [timeit.timeit(lambda: attention_forward(Q, K, V, compiled=compiled), number=1) for _ in range(benchmark_iters)]
    mean_time = sum(times) / benchmark_iters
    std_time = math.sqrt(sum((time - mean_time) ** 2 for time in times))
    return {
        "mean_time": mean_time,
        "std_time": std_time,
    }


def benchmark_backward(batch_size, seq_length, d_model, warmup_iters=5, benchmark_iters=100, compiled=False):
    Q = torch.randn(batch_size, seq_length, d_model, device="cuda", requires_grad=True)
    K = torch.randn(batch_size, seq_length, d_model, device="cuda", requires_grad=True)
    V = torch.randn(batch_size, seq_length, d_model, device="cuda", requires_grad=True)

    for _ in range(warmup_iters):
        attention_backward(Q, K, V, compiled=compiled)

    times = []
    memory = []
    for _ in range(benchmark_iters):
        t, mem = attention_backward(Q, K, V, compiled=compiled)
        times.append(t)
        memory.append(mem)

    mean_time = sum(times) / benchmark_iters
    std_time = math.sqrt(sum((time - mean_time) ** 2 for time in times))
    return {
        "mean_time": mean_time,
        "std_time": std_time,
        "memory": sum(memory) / benchmark_iters
    }


def benchmark(d_models, seq_lengths, compiled):
    benchmark_results_forward = {}
    benchmark_results_backward = {}

    for d_model in d_models:
        for seq_length in seq_lengths:
            key = f"d_{d_model}_seq_{seq_length}"
            # forward benchmarking
            try:
                result = benchmark_forward(BATCH_SIZE, seq_length, d_model, compiled=compiled)
                benchmark_results_forward[key] = result
            except Exception as e:
                print(f"Error on forward trial d_model {d_model}, seq_length {seq_length}, {e}")
                benchmark_results_forward[key] = {}

            # backward benchmarking
            try:
                result = benchmark_backward(BATCH_SIZE, seq_length, d_model, compiled=compiled)
                benchmark_results_backward[key] = result
            except Exception as e:
                print(f"Error on backward trial d_model {d_model}, seq_length {seq_length}, {e}")
                benchmark_results_backward[key] = {}
    
    return benchmark_results_forward, benchmark_results_backward
    

