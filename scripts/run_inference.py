"""
Run inference and evaluation on vision-language benchmarks.

Usage:
    python scripts/run_inference.py --config configs/baseline.yaml
    python scripts/run_inference.py --dataset mathverse --num_samples 10
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
from multiprocessing import Pool, cpu_count


def process_batch_element(args):
    """
    Worker function that preprocesses a single sample in parallel.

    Takes: (image, question, index, ground_truth, max_dimension)
    Returns: dict with preprocessed data, or None if image is missing

    Does image resizing + message dict creation.
    Note: processor.apply_chat_template() stays in main process (can't pickle processor).
    """
    image, question, idx, ground_truth, max_dimension = args

    # Skip samples without images
    if image is None:
        return None

    # Resize image if too large
    if image.width > max_dimension or image.height > max_dimension:
        image = image.copy()
        image.thumbnail((max_dimension, max_dimension))

    # Create message dictionary
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        }
    ]

    return {
        "messages": messages,
        "image": image,
        "idx": idx,
        "question": question,
        "ground_truth": ground_truth
    }


parser = argparse.ArgumentParser(description="Run inference and evaluation on VLM benchmarks")

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
print("STEM-VLM Inference")
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

# Setup multiprocessing pool for parallel preprocessing
num_workers = min(cpu_count(), args.batch_size)
print(f"✓ Using {num_workers} worker processes (batch_size={args.batch_size}, cpus={cpu_count()})")

# Create pool once, reuse for all batches (avoids repeated spawn overhead)
pool = Pool(processes=num_workers)

sample_idx = 0
num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size

for batch in tqdm(dataset.iter(batch_size=args.batch_size), desc="Evaluating", total=num_batches):
    
    # Prepare inputs for parallel processing
    batch_inputs = [
        (
            batch['image'][i],
            batch['question'][i],
            sample_idx + i,
            batch.get('answer', [None] * len(batch['image']))[i],
            MAX_DIMENSION
        )
        for i in range(len(batch['image']))
    ]

    # Process all samples in parallel across worker processes
    processed_elements = pool.map(process_batch_element, batch_inputs)

    # Filter results and apply chat template
    text_prompts = []
    images = []
    valid_indices = []
    questions = []
    ground_truths = []

    for elem in processed_elements:
        # Skip samples that had no image
        if elem is None:
            continue

        # Apply chat template (can't do this in workers - processor not picklable)
        text_prompt = processor.apply_chat_template(
            elem["messages"],
            add_generation_prompt=True
        )

        text_prompts.append(text_prompt)
        images.append(elem["image"])
        valid_indices.append(elem["idx"])
        questions.append(elem["question"])
        ground_truths.append(elem["ground_truth"])

    sample_idx += len(batch['image'])

    # Skip empty batches
    if len(text_prompts) == 0:
        continue

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
    for idx, generated_text, question, ground_truth in zip(valid_indices, generated_texts, questions, ground_truths):
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
            "ground_truth": ground_truth
        })

# Cleanup: close pool and wait for workers to finish
pool.close()  # No more tasks will be submitted
pool.join()   # Wait for all workers to terminate

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
