import torch 
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import RotaryEmbedding, TransformerBlock

device = "cuda" if torch.cuda.is_available() else "cpu"

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
x = torch.randn((4, context_length, d_model), requires_grad=True).to(device)

blocks = [TransformerBlock(
    d_model=d_model, d_ff=d_ff, num_heads=num_heads,
    positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length)
).to(device) for _ in range(16)]

def sequential_blocks(blocks, x):
    for block in blocks:
        x = block(x)
    return x

def nested_blocks_checkpoint(blocks, x):
    if len(blocks) == 1:
        return checkpoint(blocks[0], x, use_reentrant=False)
    else:
        x = checkpoint(nested_blocks_checkpoint, blocks[:len(blocks)//2], x, use_reentrant=False)
        x = checkpoint(nested_blocks_checkpoint, blocks[len(blocks)//2:], x, use_reentrant=False)
        return x

# y = nested_blocks_checkpoint(blocks, x)
y = sequential_blocks(blocks, x)
y.sum().backward()

print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / (1024**2):.2f} MiB")