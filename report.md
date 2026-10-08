## Problem: Benchmarking Script

Using the default model settings

- forward
```
Benchmarking forward mode...
Time taken: 0.020201002713292837 seconds per iteration
Variance: 4.503222838939461e-09
Standard deviation: 6.710605664870692e-05
```

- forward + backward
```
Benchmarking forward_backward mode...
Time taken: 0.07003293894231319 seconds per iteration
Variance: 3.944476621567384e-08
Standard deviation: 0.00019860706486848307
```

- forward + backward + optimizer step
```
Benchmarking forward_backward_optimizer_step mode...
Time taken: 0.07332581151276826 seconds per iteration
Variance: 8.011134572070976e-09
Standard deviation: 8.95049416069916e-05
```

The warmup is critical to reduce the variance, if we skip the warmup stage, the reported time and variance would be much higher
- forward, no warmup
```
Benchmarking forward mode...
Time taken: 0.05068178568035364 seconds per iteration
Variance: 0.008258084730262794
Standard deviation: 0.09087400470025954
```

## Problem: Nsight System Profiling

Use the same default model config as the pervous one

The forward time is around 16ms, with additional 4ms on the cuda event sync, which is aligned with the number in the above benchmarking script.

On the hardware I'm experiment with (RTX 3090, using Ampere Cuda Core), the most time consuming kernel is `ampere_sgemm_128x64_tn` kernel, which takes 33.4%, get invoked for 29 times in a single forward. In foward_backward, this kernel is still the most time consuming one, but the ratio drops to 9.6%, and the invoked time is still 29 times.

There is some other kernel taking majority of time as well. Some elementwise kernel is also taking some time, this is especially obvious in forward backward pass, for example
```
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
9.6%	6.758 ms	29	233.036 μs	124.224 μs	122.657 μs	2.004 ms	344.816 μs	ampere_sgemm_128x64_tn
8.7%	6.098 ms	68	89.680 μs	31.776 μs	1.408 μs	476.418 μs	143.101 μs	void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, std::array<char *, (unsigned long)3>>(int, T2, T3)   
```

The fraction of time spend on matrix mulitiplication drops compare a full training step vs inference only step.

(Note the following is computed over RTX 4500 which is using blackwell tensor core)
In the current profling result, if we ignore the `aten:where`, the breakdown of time is as follow:
- compute attention scores: 235.5 us
- compute softmax: 164.4 us
- compute output: 231.117 us

Compre to their FLOPs (bsz: 4, d_model: 512, d_ff: 1024, context_length: 1024, num_heads: 8, d_k: 64):
- compute attention scores: `bsz x 2 x seq x d_k x seq` = 536M
- compute softmax: `bsz x seq x seq` = 4M
- compute output: `bsz x 2 x seq x seq x d_v` = 546M

From the computation, the attention scores and output FLOPs is idential given the matrix multiplication is similar; however, softmax FLOPs is much lower given its elementwise computation property, but the nsys profiling result shows its much higher than expected.

## Problem: Mixed Precision Accumulation

The program output is as follow

```
tensor(10.0001)
tensor(9.9531, dtype=torch.float16)
tensor(10.0021)
tensor(10.0021)
```

It shows that using `torch.float16` has the underflow issue, where the cumulatived result is lower than the expected one; while represent the value in `torch.float16` and cumulative into `torch.float32` shows overflow, the cumulatived value is higher.

```
model parameters dtype: torch.float32
fc1 output dtype: torch.float16
ln output dtype: torch.float32
model prediction dtype: torch.float16
loss dtype: torch.float32
model parameters gradient dtype: torch.float32
```

LayerNorm needs to compute the norm for normalization, which is sensitive and need higher precision to avoid the cumulative error similar to the result above. Also FP16's range is too limited and could cause underflow or overflow issue. Also LayerNorm is elementwise operation and could not best leverage Tensor Cores. Change to BF16 does not resolve the issue, although its range is the save as FP32, but the precision is lower than FP16 and would still have the cumulative error.

Under the original model setup, there is no big difference in the model forward pass, and the bf16 is even slightly slower than fp32. However, once we scale up to (bsz 32, d_model 1024, d_ff 4196), ther bf16 and fp32 has relative large performance difference.

The fp32 forward pass is around 378ms, and the matrix multi kernel is taking 6.4ms on average
```
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
47.8%	180.424 ms	28	6.444 ms	2.710 ms	2.675 ms	12.386 ms	4.429 ms	void cutlass::Kernel2<cutlass_80_simt_sgemm_128x256_8x4_tn_align1>(T1::Params)
```

While for the bf16 version, the forward pass is around 164ms, and the matrix multi kernel is taking 2.2 ms on average
```
Time	Total Time	Instances	Avg	Med	Min	Max	StdDev	Name
16.1%	26.478 ms	12	2.207 ms	2.174 ms	2.136 ms	2.312 ms	69.023 μs	void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_bf16_128x256_32x3_tn_align2>(T1::Params)
```

## Problem: Memory Profiling

Command

- forward, context length 1024
`uv run python cs336_systems/benchmarking_script.py --mode forward --enable_memory_profiling true`

![forward, context length 1024](./fw_seq_1204.png)

- forward + backward + optimizer, context length 1024
`uv run python cs336_systems/benchmarking_script.py --mode forward_backward_optimizer_step --enable_memory_profiling true`

![forward backward optimizer, context length 1024](./fw_bw_optm_seq_1024.png)

Peak memory of forward: 1.9G
Peak memory of forward + backward + optimizer: 2.6G

- forward, context length 1024, mixed precision
`uv run python cs336_systems/benchmarking_script.py --mode forward --enable_memory_profiling true --dtype bfloat16`

Peak memory: 1.4G

- forward + backward + optimizer, context length 1024, mixed precision
`uv run python cs336_systems/benchmarking_script.py --mode forward_backward_optimizer_step --enable_memory_profiling true --dtype bfloat16`

Peak memory: 2.0G

Yes, the mixed precision affect the memory usage a lot, not only the compute kernel is more efficient, but the activation size is also reduced

Also, from the computation, the spike of the memory comes from the softmax computation, where we would have a activation of size `bsz x num_head x seq x seq x dtype`.

The size of the residual stream is essentially the input of shape `bsz x seq x d_model`, in the current setup with single precison, this gives `4 x 1024 x 512 x 4 = 8MB`

128MB is the largest allocation, and it comes from softmax computation.

`uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda -- python cs336_systems/benchmarking_script.py --mode forward_backward_optimizer_step --enable_memory_profiling true --dtype bfloat16`

`uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --cuda-memory-usage=true -- python cs336_systems/benchmarking_script.py --mode forward_backward_optimizer_step`

But from my trace, there is not obvious change in the memory usage during the `forward_backward_optimize` step. The memory starts to cumulative to maximize of 3GB and then stay static and not freed; my hypothesis is that this is due to Pytorch memory allocator.

`uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autograd-shapes-nvtx --cudabacktrace=all --python-backtrace=cuda --cuda-memory-usage=true -- python cs336_systems/benchmarking_script.py --mode forward_backward`

## Problem: Activation Checkpoint

Vanila way to checkpoint each block does not reduce the complexity, block 1 would depends on block 0's output and resursively. Thus although each block's intermediate activation is reduced, the overall activation is still of `O(N)`.

A better way is to use nested checkpoint, and wrap the blocks recursively to minimize the peak memory. This is similar to a balanced binary tree, where is the smallest tree hight, which is exact the peak memory (stack). This would give a `O(logN)` peak memory scale.

> PS: without checkpoint, the sequential run with 16 blocks would have CUDA OOM issue on RTX 4500

| N  | Peak memory (MiB) | Increment |
|----|-------------------|-----------|
| 4  |  9951.93          | -         |
| 8  | 13173.08          |           |
| 16 | 19615.40          |           |

If the nested checkpoint is not allowed, then we could use `sqrt(N)` blocks to warp

## Problem: PyTorch Attention

| d_model | seq_len | fwd mean (ms) | fwd std (ms) | bwd mean (ms) | bwd std (ms) | memory (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 0.350 | 0.107 | 0.731 | 1.051 | 0.0078 |
| 16 | 1024 | 0.684 | 0.044 | 1.595 | 0.804 | 0.125 |
| 16 | 4096 | 6.730 | 0.056 | 23.723 | 0.102 | 2.000 |
| 16 | 8192 | 24.843 | 0.391 | 89.011 | 0.598 | 8.000 |
| 16 | 16384 | 98.642 | 0.795 | OOM | — | OOM |
| 32 | 256 | 0.332 | 0.068 | 0.739 | 0.484 | 0.0078 |
| 32 | 1024 | 0.615 | 0.029 | 1.580 | 0.173 | 0.125 |
| 32 | 4096 | 6.944 | 0.052 | 24.141 | 0.516 | 2.000 |
| 32 | 8192 | 25.743 | 0.589 | 90.859 | 0.530 | 8.000 |
| 32 | 16384 | OOM | — | OOM | — | OOM |
| 64 | 256 | 0.335 | 0.142 | 0.707 | 0.271 | 0.0078 |
| 64 | 1024 | 0.646 | 0.039 | 1.599 | 0.106 | 0.125 |
| 64 | 4096 | 7.398 | 0.056 | 25.105 | 0.196 | 2.000 |
| 64 | 8192 | 27.548 | 0.163 | 94.530 | 0.515 | 8.000 |
| 64 | 16384 | OOM | — | OOM | — | OOM |
| 128 | 256 | 0.330 | 0.056 | 0.701 | 0.240 | 0.0078 |
| 128 | 1024 | 0.704 | 0.028 | 1.716 | 0.079 | 0.125 |
| 128 | 4096 | 8.268 | 0.059 | 26.927 | 0.239 | 2.000 |
| 128 | 8192 | 31.004 | 0.306 | 101.629 | 1.336 | 8.000 |
| 128 | 16384 | OOM | — | OOM | — | OOM |

## Problem: Torch Compile

| d_model | seq_len | fwd mean (ms) | fwd std (ms) | bwd mean (ms) | bwd std (ms) | memory (GiB) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 256 | 0.413 | 0.136 | 0.877 | 2.362 | 0.0039 |
| 16 | 1024 | 0.668 | 0.158 | 1.312 | 1.625 | 0.0625 |
| 16 | 4096 | 4.959 | 0.101 | 14.826 | 0.633 | 1.000 |
| 16 | 8192 | 18.619 | 0.595 | 58.973 | 0.848 | 4.000 |
| 16 | 16384 | 64.610 | 0.701 | 222.165 | 1.369 | 16.000 |
| 32 | 256 | 0.528 | 0.160 | 1.090 | 2.961 | 0.0039 |
| 32 | 1024 | 0.730 | 0.132 | 1.473 | 2.167 | 0.0625 |
| 32 | 4096 | 4.970 | 0.440 | 15.684 | 0.789 | 1.000 |
| 32 | 8192 | 17.174 | 0.146 | 56.571 | 0.792 | 4.000 |
| 32 | 16384 | 68.209 | 0.714 | OOM | — | OOM |
| 64 | 256 | 0.528 | 0.183 | 0.845 | 0.509 | 0.0039 |
| 64 | 1024 | 0.718 | 0.120 | 1.348 | 0.823 | 0.0625 |
| 64 | 4096 | 5.422 | 0.083 | 16.704 | 0.193 | 1.000 |
| 64 | 8192 | 18.969 | 0.127 | 60.304 | 0.617 | 4.000 |
| 64 | 16384 | 75.452 | 0.327 | OOM | — | OOM |
| 128 | 256 | 0.526 | 0.135 | 1.113 | 2.322 | 0.0039 |
| 128 | 1024 | 0.805 | 0.127 | 1.532 | 1.577 | 0.0625 |
| 128 | 4096 | 6.302 | 0.172 | 18.438 | 0.565 | 1.000 |
| 128 | 8192 | 22.434 | 0.393 | 67.308 | 0.890 | 4.000 |
| 128 | 16384 | 89.105 | 0.488 | OOM | — | OOM |

## Problem: FlashAttention-2 Benchmarking

Forward

| seq_len | d | precision | triton_ms | pytorch_ms | speedup |
|--------:|--:|----------|----------:|-----------:|--------:|
| 128 | 16 | bfloat16 | 0.009947 | 1.006232 | 101.16× |
| 128 | 16 | float32 | 0.008386 | 0.978011 | 116.62× |
| 128 | 32 | bfloat16 | 0.008061 | 0.999074 | 123.94× |
| 128 | 32 | float32 | 0.009647 | 0.982933 | 101.89× |
| 128 | 64 | bfloat16 | 0.008401 | 1.002523 | 119.33× |
| 128 | 64 | float32 | 0.012284 | 1.030720 | 83.91× |
| 128 | 128 | bfloat16 | 0.009036 | 1.004431 | 111.16× |
| 128 | 128 | float32 | 0.019919 | 0.987418 | 49.57× |
| 256 | 16 | bfloat16 | 0.008032 | 3.409719 | 424.52× |
| 256 | 16 | float32 | 0.009075 | 3.352167 | 369.38× |
| 256 | 32 | bfloat16 | 0.008460 | 3.355569 | 396.64× |
| 256 | 32 | float32 | 0.012886 | 3.346690 | 259.71× |
| 256 | 64 | bfloat16 | 0.009347 | 3.353878 | 358.82× |
| 256 | 64 | float32 | 0.016171 | 3.414295 | 211.14× |
| 256 | 128 | bfloat16 | 0.010658 | 3.353288 | 314.63× |
| 256 | 128 | float32 | 0.027386 | 3.334513 | 121.76× |
| 512 | 16 | bfloat16 | 0.010478 | 12.392940 | 1182.76× |
| 512 | 16 | float32 | 0.012212 | 12.309040 | 1007.95× |
| 512 | 32 | bfloat16 | 0.010596 | 12.285576 | 1159.45× |
| 512 | 32 | float32 | 0.019041 | 12.258080 | 643.77× |
| 512 | 64 | bfloat16 | 0.012564 | 12.253348 | 975.27× |
| 512 | 64 | float32 | 0.024738 | 12.488133 | 504.82× |
| 512 | 128 | bfloat16 | 0.014510 | 12.689824 | 874.56× |
| 512 | 128 | float32 | 0.042845 | 12.934852 | 301.90× |
| 1024 | 16 | bfloat16 | 0.014340 | 47.098558 | 3284.42× |
| 1024 | 16 | float32 | 0.017874 | 47.157425 | 2638.33× |
| 1024 | 32 | bfloat16 | 0.015158 | 46.713154 | 3081.75× |
| 1024 | 32 | float32 | 0.031683 | 46.875904 | 1479.53× |
| 1024 | 64 | bfloat16 | 0.018444 | 46.769360 | 2535.75× |
| 1024 | 64 | float32 | 0.044207 | 48.008816 | 1086.00× |
| 1024 | 128 | bfloat16 | 0.022168 | 47.168911 | 2127.79× |
| 1024 | 128 | float32 | 0.073099 | 46.632784 | 638.00× |
| 2048 | 16 | bfloat16 | 0.022928 | 182.094940 | 7942.03× |
| 2048 | 16 | float32 | 0.029602 | 185.226974 | 6257.25× |
| 2048 | 32 | bfloat16 | 0.024130 | 185.690140 | 7695.41× |
| 2048 | 32 | float32 | 0.057029 | 182.367737 | 3197.81× |
| 2048 | 64 | bfloat16 | 0.030597 | 182.300507 | 5958.12× |
| 2048 | 64 | float32 | 0.078785 | 185.160736 | 2350.20× |
| 2048 | 128 | bfloat16 | 0.037370 | 181.826553 | 4865.58× |
| 2048 | 128 | float32 | 0.134641 | 179.991196 | 1336.82× |
| 4096 | 16 | bfloat16 | 0.039648 | 733.253479 | 18494.09× |
| 4096 | 16 | float32 | 0.053197 | 731.898193 | 13758.26× |
| 4096 | 32 | bfloat16 | 0.042169 | 716.027405 | 16979.95× |
| 4096 | 32 | float32 | 0.107623 | 727.885193 | 6763.29× |
| 4096 | 64 | bfloat16 | 0.054848 | 731.423340 | 13335.46× |
| 4096 | 64 | float32 | 0.142145 | 726.201477 | 5108.88× |
| 4096 | 128 | bfloat16 | 0.068261 | 716.130737 | 10491.07× |
| 4096 | 128 | float32 | 0.248542 | 770.930359 | 3101.81× |

Backward

| seq_len | d | precision | triton_ms | pytorch_ms | speedup |
|--------:|--:|----------|----------:|-----------:|--------:|
| 128 | 16 | bfloat16 | 0.245261 | 2.424834 | 9.89× |
| 128 | 16 | float32 | 0.234723 | 2.385765 | 10.16× |
| 128 | 32 | bfloat16 | 0.232184 | 2.385127 | 10.27× |
| 128 | 32 | float32 | 0.234848 | 2.388786 | 10.17× |
| 128 | 64 | bfloat16 | 0.254194 | 2.452666 | 9.65× |
| 128 | 64 | float32 | 0.232183 | 2.418572 | 10.42× |
| 128 | 128 | bfloat16 | 0.238334 | 2.411037 | 10.12× |
| 128 | 128 | float32 | 0.233919 | 2.426480 | 10.37× |
| 256 | 16 | bfloat16 | 0.231078 | 8.513958 | 36.84× |
| 256 | 16 | float32 | 0.231629 | 8.453396 | 36.50× |
| 256 | 32 | bfloat16 | 0.231829 | 8.531145 | 36.80× |
| 256 | 32 | float32 | 0.233292 | 8.564672 | 36.71× |
| 256 | 64 | bfloat16 | 0.231600 | 8.640492 | 37.31× |
| 256 | 64 | float32 | 0.233860 | 8.740215 | 37.37× |
| 256 | 128 | bfloat16 | 0.234852 | 8.908186 | 37.93× |
| 256 | 128 | float32 | 0.232410 | 8.642051 | 37.18× |
| 512 | 16 | bfloat16 | 0.231679 | 33.379375 | 144.08× |
| 512 | 16 | float32 | 0.231957 | 33.408000 | 144.03× |
| 512 | 32 | bfloat16 | 0.233613 | 32.621652 | 139.64× |
| 512 | 32 | float32 | 0.234393 | 32.270986 | 137.68× |
| 512 | 64 | bfloat16 | 0.233592 | 32.463125 | 138.97× |
| 512 | 64 | float32 | 0.236131 | 33.201218 | 140.61× |
| 512 | 128 | bfloat16 | 0.233380 | 33.233007 | 142.40× |
| 512 | 128 | float32 | 0.234569 | 32.956976 | 140.50× |
| 1024 | 16 | bfloat16 | 0.233521 | 130.605026 | 559.29× |
| 1024 | 16 | float32 | 0.234296 | 131.057220 | 559.37× |
| 1024 | 32 | bfloat16 | 0.244181 | 130.546661 | 534.63× |
| 1024 | 32 | float32 | 0.244173 | 130.929733 | 536.22× |
| 1024 | 64 | bfloat16 | 0.235175 | 132.051300 | 561.50× |
| 1024 | 64 | float32 | 0.237259 | 131.884064 | 555.86× |
| 1024 | 128 | bfloat16 | 0.253929 | 132.171997 | 520.51× |
| 1024 | 128 | float32 | 0.254846 | 129.274078 | 507.26× |
| 2048 | 16 | bfloat16 | 0.234806 | 524.439392 | 2233.50× |
| 2048 | 16 | float32 | 0.261772 | 522.409546 | 1995.67× |
| 2048 | 32 | bfloat16 | 0.240164 | 520.874146 | 2168.83× |
| 2048 | 32 | float32 | 0.250533 | 514.931580 | 2055.34× |
| 2048 | 64 | bfloat16 | 0.251609 | 511.179352 | 2031.64× |
| 2048 | 64 | float32 | 0.255616 | 511.869781 | 2002.49× |
| 2048 | 128 | bfloat16 | 0.366972 | 510.877014 | 1392.14× |
| 2048 | 128 | float32 | 0.479428 | 512.785950 | 1069.58× |
| 4096 | 16 | bfloat16 | 0.234108 | 2073.045898 | 8855.08× |
| 4096 | 16 | float32 | 0.253061 | 2064.981934 | 8159.99× |
| 4096 | 32 | bfloat16 | 0.253542 | 2092.986816 | 8254.99× |
| 4096 | 32 | float32 | 0.280402 | 2037.889648 | 7267.74× |
| 4096 | 64 | bfloat16 | 0.317580 | 2045.781494 | 6441.78× |
| 4096 | 64 | float32 | 0.414678 | 2052.740723 | 4950.20× |
| 4096 | 128 | bfloat16 | 0.743516 | 2047.616577 | 2753.97× |
| 4096 | 128 | float32 | 0.952795 | 2050.346191 | 2151.93× |

Forward + Backward

| seq_len | d | precision | triton_ms | pytorch_ms | speedup |
|--------:|--:|----------|----------:|-----------:|--------:|
| 128 | 16 | bfloat16 | 0.184765 | 2.539259 | 13.74× |
| 128 | 16 | float32 | 0.202985 | 2.729070 | 13.44× |
| 128 | 32 | bfloat16 | 0.201007 | 2.586894 | 12.87× |
| 128 | 32 | float32 | 0.211539 | 2.588771 | 12.24× |
| 128 | 64 | bfloat16 | 0.221453 | 2.579437 | 11.65× |
| 128 | 64 | float32 | 0.207006 | 2.749324 | 13.28× |
| 128 | 128 | bfloat16 | 0.215249 | 2.625900 | 12.20× |
| 128 | 128 | float32 | 0.216078 | 2.584679 | 11.96× |
| 256 | 16 | bfloat16 | 0.200891 | 8.010301 | 39.87× |
| 256 | 16 | float32 | 0.194087 | 7.998112 | 41.21× |
| 256 | 32 | bfloat16 | 0.210059 | 8.074965 | 38.44× |
| 256 | 32 | float32 | 0.216973 | 8.582641 | 39.56× |
| 256 | 64 | bfloat16 | 0.202593 | 8.097776 | 39.97× |
| 256 | 64 | float32 | 0.211468 | 8.263899 | 39.08× |
| 256 | 128 | bfloat16 | 0.209034 | 7.977336 | 38.16× |
| 256 | 128 | float32 | 0.203639 | 8.131019 | 39.93× |
| 512 | 16 | bfloat16 | 0.207994 | 30.833814 | 148.24× |
| 512 | 16 | float32 | 0.211378 | 29.291349 | 138.57× |
| 512 | 32 | bfloat16 | 0.208405 | 28.825994 | 138.32× |
| 512 | 32 | float32 | 0.194281 | 29.066752 | 149.61× |
| 512 | 64 | bfloat16 | 0.200846 | 28.233611 | 140.57× |
| 512 | 64 | float32 | 0.205479 | 27.942400 | 135.99× |
| 512 | 128 | bfloat16 | 0.217555 | 26.674891 | 122.61× |
| 512 | 128 | float32 | 0.219060 | 28.019573 | 127.91× |
| 1024 | 16 | bfloat16 | 0.202680 | 104.957535 | 517.85× |
| 1024 | 16 | float32 | 0.204462 | 120.120384 | 587.49× |
| 1024 | 32 | bfloat16 | 0.205424 | 113.998146 | 554.94× |
| 1024 | 32 | float32 | 0.201419 | 108.540062 | 538.88× |
| 1024 | 64 | bfloat16 | 0.208146 | 109.561310 | 526.37× |
| 1024 | 64 | float32 | 0.212886 | 121.967552 | 572.92× |
| 1024 | 128 | bfloat16 | 0.221329 | 116.799965 | 527.72× |
| 1024 | 128 | float32 | 0.301294 | 115.391037 | 382.98× |
| 2048 | 16 | bfloat16 | 0.209144 | 420.283112 | 2009.54× |
| 2048 | 16 | float32 | 0.210212 | 481.711853 | 2291.55× |
| 2048 | 32 | bfloat16 | 0.201506 | 413.763123 | 2053.35× |
| 2048 | 32 | float32 | 0.219746 | 432.558319 | 1968.45× |
| 2048 | 64 | bfloat16 | 0.213316 | 422.450531 | 1980.40× |
| 2048 | 64 | float32 | 0.277084 | 474.280029 | 1711.68× |
| 2048 | 128 | bfloat16 | 0.419988 | 415.264313 | 988.75× |
| 2048 | 128 | float32 | 0.579603 | 422.501892 | 728.95× |
| 4096 | 16 | bfloat16 | 0.220214 | 1662.922363 | 7551.39× |
| 4096 | 16 | float32 | 0.230706 | 1714.981201 | 7433.62× |
| 4096 | 32 | bfloat16 | 0.224194 | 1714.519165 | 7647.48× |
| 4096 | 32 | float32 | 0.350429 | 1680.558350 | 4795.72× |
| 4096 | 64 | bfloat16 | 0.365693 | 1669.322266 | 4564.82× |
| 4096 | 64 | float32 | 0.524722 | 1700.472046 | 3240.71× |
| 4096 | 128 | bfloat16 | 0.825465 | 1697.452026 | 2056.36× |
| 4096 | 128 | float32 | 1.136427 | 1675.357910 | 1474.23× |
