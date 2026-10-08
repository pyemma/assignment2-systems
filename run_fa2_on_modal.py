import modal

image = modal.Image.debian_slim(python_version="3.12") \
    .uv_pip_install("torch", "numpy", "einops", "jaxtyping", "pytest", "triton", "pandas") \
    .add_local_python_source("cs336_systems", "cs336_basics") \
    .add_local_dir("tests", "/root/tests")

app = modal.App(image=image)

# @app.function(gpu="A100")
# def run():
#     import subprocess

#     completed = subprocess.run(
#         ["pytest", "tests", "-k", "test_flash_forward_pass_triton"],
#         cwd="/root",
#         check=False,
#     )

#     if completed.returncode != 0:
#         raise SystemExit(completed.returncode)

# @app.function(gpu="A100")
# def run():
#     import subprocess

#     completed = subprocess.run(
#         ["pytest", "tests", "-k", "test_flash_backward"],
#         cwd="/root",
#         check=False
#     )

#     if completed.returncode != 0:
#         raise SystemExit(completed.returncode)

@app.function(gpu="H100", timeout=1200)
def run():
    from cs336_systems.benchmark_fa2 import benchmark

    df1 = benchmark("fwd_bwd", "triton")
    df2 = benchmark("fwd_bwd", "pytorch")

    print(df1)
    print(df2)