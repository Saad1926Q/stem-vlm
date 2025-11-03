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


parser = argparse.ArgumentParser(description="Run baseline evaluation")

# Config file
parser.add_argument('--config', type=str, help='Path to YAML config file')

# Model args
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])

# Dataset args
parser.add_argument('--dataset', type=str, default='mathvista', choices=['mathvista', 'scienceqa'])
parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (None=all)')

# Generation args
parser.add_argument('--max_new_tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.0)

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
    if 'output' in config:
        args.output_dir = config['output'].get('dir', args.output_dir)

print("=" * 70)
print("STEM-VLM Baseline Evaluation")
print("=" * 70)
print(f"Model: {args.model_name}")
print(f"Dataset: {args.dataset}")
print(f"Num samples: {args.num_samples if args.num_samples else 'all'}")
print(f"Output: {args.output_dir}")
print("=" * 70)



# Load processor (handles image + text preprocessing)
processor = AutoProcessor.from_pretrained(args.model_name)

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
    device_map="auto"
)
model.eval()  

print(f" Model loaded on {model.device}")
if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"  GPU memory: {mem_gb:.2f} GB")


if args.dataset == 'mathvista':
    dataset = load_dataset("AI4Math/MathVista", split="test")
elif args.dataset == 'scienceqa':
    dataset = load_dataset("derek-thomas/ScienceQA", split="test")

if args.num_samples:
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))

print(f"✓ Loaded {len(dataset)} samples")



results = []

for idx, sample in enumerate(tqdm(dataset, desc="Evaluating")):

    # Get image and question from sample
    image = sample['image']
    question = sample['question']

    # Format as Qwen2-VL expects (conversation format)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process inputs (convert to tensors)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )

    # Move to GPU/CPU
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate answer
    # torch.no_grad(): Don't compute gradients (saves memory)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=(args.temperature > 0)  # Only sample if temp > 0
        )

    # Decode output tokens back to text
    generated_text = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    # Extract just the answer (after "assistant\n")
    if "assistant\n" in generated_text:
        prediction = generated_text.split("assistant\n")[-1].strip()
    else:
        prediction = generated_text.strip()

    # Save result
    results.append({
        "sample_id": idx,
        "question": question,
        "prediction": prediction,
        "ground_truth": sample.get("answer", None)
    })

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
