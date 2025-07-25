from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-32B-Instruct",  # Official quantized HF version for 32B
    local_dir="./models/Qwen2.5-VL-32B-Instruct",
    local_dir_use_symlinks=False
)
