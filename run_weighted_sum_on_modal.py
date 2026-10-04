import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("torch", "triton", "einops")
    .add_local_python_source("cs336_systems")
)

app = modal.App(image=image)


@app.function(gpu="A100")
def run():
    import torch
    from cs336_systems.triton_weighted_sum import f_weightedsum

    x = torch.rand((32, 64), device="cuda", requires_grad=True)
    weight = torch.rand((64,), device="cuda", requires_grad=True)
    y = f_weightedsum(x, weight)
    print("forward", tuple(y.shape))
    print(y)

    y.sum().backward()
    print("grad_x", tuple(x.grad.shape))
    print("grad_weight", tuple(weight.grad.shape))
