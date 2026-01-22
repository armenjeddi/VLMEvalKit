from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="OpenGVLab/InternVL3_5-8B",
    local_dir="./internvl",
    local_dir_use_symlinks=False,
    ignore_patterns=["*.py"]
)