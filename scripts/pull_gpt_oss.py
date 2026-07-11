import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from thinruntime.model_cache import cached_model_path

def main():
    model_id = "openai/gpt-oss-20b"
    target_dir = cached_model_path(model_id)
    print(f"Starting filtered pull for {model_id} → {target_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(target_dir),
        ignore_patterns=["original/*", "metal/*"],
    )
    print("Download completed successfully!")

if __name__ == "__main__":
    main()
