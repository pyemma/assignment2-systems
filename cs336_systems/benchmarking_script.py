import argparse
import logging
import timeit
from typing import Callable

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

VOCAB_SIZE = 10_000
logger = logging.getLogger(__name__)

def forward_pass(model, input) -> None:
    model(input)
    torch.cuda.synchronize()


def forward_backward_pass(model, input, target) -> None:
    output = model(input)
    loss = cross_entropy(output, target)
    loss.backward()
    torch.cuda.synchronize()


def forward_backward_optimizer_step(model, input, target, optimizer) -> None:
    output = model(input)
    loss = cross_entropy(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()


BENCHMARK_FUNCTIONS = {
    "forward": forward_pass,
    "forward_backward": forward_backward_pass,
    "forward_backward_optimizer_step": forward_backward_optimizer_step,
}

def generate_random_batch(
    vocab_size: int,
    batch_size: int,
    context_length: int,
) -> torch.Tensor:
    return torch.randint(vocab_size, (batch_size, context_length))


def main(
    context_length: int = 1024,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    d_ff: int = 1024,
    batch_size: int = 4,
    warmup_iters: int = 5,
    benchmark_iters: int = 10,
    mode: str = "forward",
):

    # setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # construct model arguments
    model_args = {
        "vocab_size": VOCAB_SIZE,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
    }

    model = BasicsTransformerLM(**model_args).to(device)

    # setup optimizer
    optimizer = AdamW(model.parameters())

    # generate random batch of data
    batch = generate_random_batch(VOCAB_SIZE, batch_size, context_length + 1).to(device)
    input = batch[:, :-1]
    target = batch[:, 1:]

    func = BENCHMARK_FUNCTIONS[mode]
    params = {"model": model, "input": input}
    if mode in ["forward_backward", "forward_backward_optimizer_step"]:
        params["target"] = target
    if mode == "forward_backward_optimizer_step":
        params["optimizer"] = optimizer

    
    # warmup before benchmarking
    for _ in range(warmup_iters):
        func(**params)

    # benchmark
    t = timeit.timeit(func, **params, number=benchmark_iters)
    logger.info(f"Time taken: {t/benchmark_iters} seconds per iteration")

if __name__ == "__main__":
    main()