import modal

image = modal.Image.debian_slim(python_version="3.12") \
    .uv_pip_install("torch", "numpy", "einops", "jaxtyping", "pytest", "triton") \
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

@app.function(gpu="A100")
def run():
    import subprocess

    completed = subprocess.run(
        ["pytest", "tests", "-k", "test_flash_backward"],
        cwd="/root",
        check=False
    )

    if completed.returncode != 0:
        raise SystemExit(completed.returncode)