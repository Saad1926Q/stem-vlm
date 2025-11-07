"""
Compare models from WandB Model Registry.

Usage:
    # Compare all models (shows artifact names and accuracy if available)
    python scripts/compare_models.py

    # Compare all Qwen2-VL-2B models
    python scripts/compare_models.py --tags Qwen2-VL-2B-Instruct

    # Compare models from lora-rank-sweep experiment with specific columns
    python scripts/compare_models.py --tags lora-rank-sweep --columns artifact_name lora_r learning_rate final_train_loss accuracy

    # Export to CSV
    python scripts/compare_models.py --tags baseline --output comparison.csv
"""

import argparse
import wandb
from pathlib import Path
import pandas as pd
from tabulate import tabulate


parser = argparse.ArgumentParser(description="Compare models from WandB Model Registry")

parser.add_argument('--wandb_project', type=str, default='stem-vlm-training',
                    help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='WandB entity/username')
parser.add_argument('--tags', type=str, nargs='*', default=None,
                    help='Filter by tags (multiple tags = AND condition)')
parser.add_argument('--columns', type=str, nargs='*', default=None,
                    help='Columns to display (if not specified, shows artifact_name only)')
parser.add_argument('--sort_by', type=str, default='training_date',
                    help='Metadata field to sort by (e.g., "final_train_loss", "lora_r", "training_date")')
parser.add_argument('--ascending', action='store_true', default=False,
                    help='Sort in ascending order (default: descending)')
parser.add_argument('--output', type=str, default=None,
                    help='Output CSV file path (optional)')
parser.add_argument('--limit', type=int, default=None,
                    help='Limit number of models to display')

args = parser.parse_args()

print("=" * 70)
print("Model Registry Comparison")
print("=" * 70)
print(f"Project: {args.wandb_project}")
if args.tags:
    print(f"Filtering by tags: {', '.join(args.tags)}")
print("=" * 70)

# Initialize WandB API
api = wandb.Api()

# Build artifact collection path
if args.wandb_entity:
    collection_path = f"{args.wandb_entity}/{args.wandb_project}"
else:
    collection_path = args.wandb_project

# Fetch all model artifacts
print(f"\nFetching models from {collection_path}...")
artifacts = api.artifacts(type_name="model", name=collection_path)

# Collect model data
models_data = []

for artifact in artifacts:
    # Filter by tags if specified
    if args.tags:
        artifact_aliases = set(artifact.aliases)
        # Check if all requested tags are present (AND condition)
        if not all(tag in artifact_aliases for tag in args.tags):
            continue

    metadata = artifact.metadata

    # Build row with artifact info and metadata
    row = {
        'artifact_name': artifact.name,
        'version': artifact.version,
        **metadata  # Unpack all metadata fields
    }

    models_data.append(row)

print(f"✓ Found {len(models_data)} models")

if len(models_data) == 0:
    print("\nNo models found matching the criteria.")
    exit(0)

# Convert to DataFrame for easy manipulation
df = pd.DataFrame(models_data)

# Sort by specified column
if args.sort_by in df.columns:
    df = df.sort_values(by=args.sort_by, ascending=args.ascending)
else:
    print(f"\nWarning: '{args.sort_by}' not found in metadata. Available columns: {', '.join(df.columns)}")

# Limit results if specified
if args.limit:
    df = df.head(args.limit)

# Display table
print("\n" + "=" * 70)
print("Model Comparison")
print("=" * 70)

# Determine which columns to display
if args.columns:
    # User specified columns
    display_columns = [col for col in args.columns if col in df.columns]
    missing_columns = [col for col in args.columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: These columns not found: {', '.join(missing_columns)}")
else:
    # Default: show artifact name and dataset-specific accuracies (if available)
    display_columns = ['artifact_name']
    if 'accuracy_scienceqa' in df.columns:
        display_columns.append('accuracy_scienceqa')
    if 'accuracy_mathverse' in df.columns:
        display_columns.append('accuracy_mathverse')

print(tabulate(df[display_columns], headers='keys', tablefmt='grid', showindex=False))

# Save to CSV if requested
if args.output:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison saved to: {output_path}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
