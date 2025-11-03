"""
Run baseline evaluation on MathVista or ScienceQA.

Usage:
    python scripts/run_baseline.py --config configs/baseline.yaml
    python scripts/run_baseline.py --dataset mathvista --num_samples 10
"""

import argparse
import yaml
import torch
import json
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from PIL import Image


parser = argparse.ArgumentParser(description="Run baseline evaluation")

# Config file
parser.add_argument('--config', type=str, help='Path to YAML config file')

# Model args
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])

# Dataset args
parser.add_argument('--dataset', type=str, default='mathverse', choices=['mathverse', 'scienceqa'])
parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (None=all)')

# Generation args
parser.add_argument('--max_new_tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')

# Output
parser.add_argument('--output_dir', type=str, default='experiments/baseline')

args = parser.parse_args()

# Load config file if provided
if args.config:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with config values (command-line args take priority)
    if 'model' in config:
        args.model_name = config['model'].get('name', args.model_name)
        args.dtype = config['model'].get('dtype', args.dtype)
    if 'dataset' in config:
        args.dataset = config['dataset'].get('name', args.dataset)
        args.num_samples = config['dataset'].get('num_samples', args.num_samples)
    if 'generation' in config:
        args.max_new_tokens = config['generation'].get('max_new_tokens', args.max_new_tokens)
        args.temperature = config['generation'].get('temperature', args.temperature)
        args.batch_size = config['generation'].get('batch_size', args.batch_size)
    if 'output' in config:
        args.output_dir = config['output'].get('dir', args.output_dir)

print("=" * 70)
print("STEM-VLM Baseline Evaluation")
print("=" * 70)
print(f"Model: {args.model_name}")
print(f"Dataset: {args.dataset}")
print(f"Num samples: {args.num_samples if args.num_samples else 'all'}")
print(f"Batch size: {args.batch_size}")
print(f"Output: {args.output_dir}")
print("=" * 70)



# Load processor (handles image + text preprocessing)
processor = AutoProcessor.from_pretrained(
    args.model_name,
    trust_remote_code=True
)

dtype_map = {
    'bfloat16': torch.bfloat16,
    'float16': torch.float16,
    'float32': torch.float32,
}
dtype = dtype_map[args.dtype]

# Load model
# device_map="auto": Automatically use GPU if available
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)
model.eval()  

print(f" Model loaded on {model.device}")
if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory: {mem_gb:.2f} GB")


if args.dataset == 'mathverse':
    dataset = load_dataset("AI4Math/MathVerse", "testmini")['testmini']
elif args.dataset == 'scienceqa':
    dataset = load_dataset("derek-thomas/ScienceQA", split="test")

if args.num_samples:
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))

print(f"✓ Loaded {len(dataset)} samples")



results = []

MAX_DIMENSION = 512

# Process dataset in batches
batch_samples = []
batch_indices = []

for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):
    # Get image and question from sample
    image = sample.get('image')
    question = sample['question']

    # Skip text-only questions (no image)
    if image is None:
        print(f"Skipping sample {idx}: no image")
        continue

    # Resize large images
    if image.width > MAX_DIMENSION or image.height > MAX_DIMENSION:
        image.thumbnail((MAX_DIMENSION, MAX_DIMENSION))

    # Add to batch
    batch_samples.append({
        'image': image,
        'question': question,
        'ground_truth': sample.get("answer", None)
    })
    batch_indices.append(idx)

    # Process batch when full (or last batch)
    if len(batch_samples) == args.batch_size or idx == len(dataset) - 1:
        # Format all samples in batch
        text_prompts = []
        images = []

        for batch_sample in batch_samples:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": batch_sample['image']},
                        {"type": "text", "text": batch_sample['question']}
                    ]
                }
            ]
            text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            text_prompts.append(text_prompt)
            images.append(batch_sample['image'])

        # Process all inputs together
        inputs = processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        # Move to GPU/CPU
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate answers for entire batch
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=(args.temperature > 0)
            )

        # Decode all outputs
        generated_texts = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Process each output in the batch
        for batch_idx, generated_text in enumerate(generated_texts):
            # Extract just the answer (after "assistant\n")
            if "assistant\n" in generated_text:
                prediction = generated_text.split("assistant\n")[-1].strip()
            else:
                prediction = generated_text.strip()

            # Save result
            results.append({
                "sample_id": batch_indices[batch_idx],
                "question": batch_samples[batch_idx]['question'],
                "prediction": prediction,
                "ground_truth": batch_samples[batch_idx]['ground_truth']
            })

        # Clear batch
        batch_samples = []
        batch_indices = []

print(f"✓ Evaluated {len(results)} samples")


# Create output directory
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save predictions to JSON
predictions_file = output_dir / f"{args.dataset}_predictions_{timestamp}.json"
with open(predictions_file, 'w') as f:
    json.dump({
        "metadata": {
            "model": args.model_name,
            "dataset": args.dataset,
            "num_samples": len(results),
            "timestamp": timestamp,
            "dtype": args.dtype,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "batch_size": args.batch_size,
        },
        "predictions": results
    }, f, indent=2)

print(f"✓ Predictions saved to: {predictions_file}")

# Save config used for this run
config_file = output_dir / f"config_{timestamp}.yaml"
with open(config_file, 'w') as f:
    yaml.dump(vars(args), f)
print(f"✓ Config saved to: {config_file}")

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
