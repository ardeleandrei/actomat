from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",  # or your desired model
    local_dir="./models/Llama-3.1-8B-Instruct",
    local_dir_use_symlinks=False   # Set to False to get real copies, not symlinks
)
