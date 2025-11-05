"""
Preprocess ScienceQA dataset for training and save as pickle files.

Usage:
    python data/preprocess.py
    python data/preprocess.py --max_samples 1000
"""

import argparse
import pickle
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


def format_sample(sample):
    """
    Convert a single sample to Unsloth training format.

    Returns dict with:
        - messages: chat messages structure (for UnslothVisionDataCollator)
        - question: original question (for reference)
        - answer: original answer (for reference)
    """
    image = sample.get('image')
    question = sample.get('question', '')
    answer = sample.get('answer', '')

    # Skip if missing required fields
    if image is None or not question or not answer:
        return None

    # Create messages structure that UnslothVisionDataCollator expects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(answer)}
            ]
        }
    ]

    return {
        "messages": messages,
        "question": question,
        "answer": str(answer)
    }


parser = argparse.ArgumentParser(description="Preprocess ScienceQA for training")

parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to process (None = all)')
parser.add_argument('--output_dir', type=str, default='data/processed',
                    help='Output directory for pickle files')

args = parser.parse_args()

print("=" * 70)
print("ScienceQA Data Preprocessing")
print("=" * 70)
print(f"Max samples: {args.max_samples if args.max_samples else 'all'}")
print("=" * 70)

print("\nLoading dataset...")
dataset = load_dataset("derek-thomas/ScienceQA")
print("Dataset loaded")

print("\nProcessing train split...")
train_data = dataset['train']
print(f"Loaded {len(train_data)} training samples")

if args.max_samples and len(train_data) > args.max_samples:
    train_data = train_data.select(range(args.max_samples))
    print(f"Limited to {len(train_data)} samples")

processed_train = []
skipped_train = 0

for sample in tqdm(train_data, desc="Formatting train"):
    formatted = format_sample(sample)
    if formatted is not None:
        processed_train.append(formatted)
    else:
        skipped_train += 1

print(f"Processed: {len(processed_train)} samples")
if skipped_train > 0:
    print(f"Skipped: {skipped_train} samples (missing image/question/answer)")

# Process validation split if it exists
processed_val = None
if 'validation' in dataset:
    print("\nProcessing validation split...")
    val_data = dataset['validation']
    print(f"Loaded {len(val_data)} validation samples")

    if args.max_samples and len(val_data) > args.max_samples:
        val_data = val_data.select(range(args.max_samples))
        print(f"Limited to {len(val_data)} samples")

    processed_val = []
    skipped_val = 0

    for sample in tqdm(val_data, desc="Formatting validation"):
        formatted = format_sample(sample)
        if formatted is not None:
            processed_val.append(formatted)
        else:
            skipped_val += 1

    print(f"Processed: {len(processed_val)} samples")
    if skipped_val > 0:
        print(f"Skipped: {skipped_val} samples (missing image/question/answer)")

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Save train pickle file
train_filename = "scienceqa_train.pkl"
if args.max_samples:
    train_filename = f"scienceqa_train_{args.max_samples}.pkl"

train_path = output_dir / train_filename
print(f"\nSaving train data to: {train_path}")
with open(train_path, 'wb') as f:
    pickle.dump(processed_train, f)
print(f"Saved {len(processed_train)} training samples")

# Save validation if it exists
if processed_val is not None:
    val_filename = "scienceqa_validation.pkl"
    if args.max_samples:
        val_filename = f"scienceqa_validation_{args.max_samples}.pkl"

    val_path = output_dir / val_filename
    print(f"\nSaving validation data to: {val_path}")
    with open(val_path, 'wb') as f:
        pickle.dump(processed_val, f)
    print(f"Saved {len(processed_val)} validation samples")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
print(f"\nTo use in training:")
print(f"  python scripts/train.py --train_data {train_path}")
if processed_val is not None:
    print(f"  python scripts/train.py --train_data {train_path} --val_data {val_path}")
