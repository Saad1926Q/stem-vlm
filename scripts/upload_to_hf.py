"""
Upload trained model to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --model_path experiments/runs/my-run/checkpoint-final --repo_name username/model-name
    python scripts/upload_to_hf.py --model_path models/my-model --repo_name username/model-name --private
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo


parser = argparse.ArgumentParser(description="Upload model to Hugging Face Hub")

parser.add_argument('--model_path', type=str, required=True,
                    help='Path to model directory (contains adapter_model.safetensors or model.safetensors)')
parser.add_argument('--repo_name', type=str, required=True,
                    help='Repository name on HuggingFace Hub (format: username/repo-name)')
parser.add_argument('--private', action='store_true', default=False,
                    help='Make repository private')
parser.add_argument('--commit_message', type=str, default='Upload model',
                    help='Commit message for the upload')

args = parser.parse_args()

model_path = Path(args.model_path)

if not model_path.exists():
    print(f"Error: Model path does not exist: {model_path}")
    exit(1)

if not model_path.is_dir():
    print(f"Error: Model path must be a directory: {model_path}")
    exit(1)




# Create repository if it doesn't exist
print(f"\nCreating repository '{args.repo_name}'...")
try:
    create_repo(
        repo_id=args.repo_name,
        private=args.private,
        exist_ok=True,
    )
    print("Repository ready")
except Exception as e:
    print(f"Error creating repository: {e}")
    exit(1)

# Upload model files
print(f"\nUploading model from {model_path}...")
api = HfApi()

try:
    api.upload_folder(
        folder_path=str(model_path),
        repo_id=args.repo_name,
        commit_message=args.commit_message,
    )
    print("Upload complete!")
except Exception as e:
    print(f"Error uploading model: {e}")
    exit(1)

print("\n" + "=" * 70)
print(f"\nYour model is now available at:")
print(f"  https://huggingface.co/{args.repo_name}")