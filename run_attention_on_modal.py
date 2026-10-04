import pprint
import modal

image = modal.Image.debian_slim(python_version="3.12") \
    .uv_pip_install("torch", "numpy", "einops", "jaxtyping") \
    .add_local_python_source("cs336_systems", "cs336_basics")

app = modal.App(image=image)

@app.function(gpu="A100")
def run():
    from cs336_systems.benchmark_attention import benchmark

    forward, backward = benchmark(
        d_models=[16, 32, 64, 128], 
        seq_lengths=[256, 1024, 4096, 8192, 16384],
        compiled=True
    )

    pprint.pprint(forward)
    print("========")
    pprint.pprint(backward)