"""
Run inference and evaluation on vision-language benchmarks.

Usage Examples:

1. Baseline inference (no adapter):
    python scripts/run_inference.py --dataset mathverse --use_wandb

2. Using WandB artifact (just artifact name):
    python scripts/run_inference.py --adapter_artifact my-model:v0 --dataset scienceqa --use_wandb

3. Using WandB artifact (with project):
    python scripts/run_inference.py --adapter_artifact stem-vlm/my-model:latest --dataset mathverse

4. Using WandB artifact (full path with entity):
    python scripts/run_inference.py --adapter_artifact username/stem-vlm/my-model:v0 --dataset scienceqa

5. Limited samples for testing:
    python scripts/run_inference.py --adapter_artifact my-model:v0 --dataset mathverse --num_samples 10

6. With Chain-of-Thought reasoning:
    python scripts/run_inference.py --adapter_artifact my-model:v0 --dataset scienceqa \
        --cot_instruction "Think step by step and explain your reasoning before selecting your final answer."

7. Using config file:
    python scripts/run_inference.py --config configs/inference.yaml

8. Config file with command-line overrides:
    python scripts/run_inference.py --config configs/inference.yaml --num_samples 100 --batch_size 8

9. Custom WandB settings:
    python scripts/run_inference.py --adapter_artifact my-model:v0 --dataset mathverse \
        --use_wandb --wandb_entity my-team --wandb_run_name custom-inference-run

10. Different model and batch size:
    python scripts/run_inference.py --model_name Qwen/Qwen2-VL-7B-Instruct \
        --adapter_artifact my-7b-model:v0 --dataset scienceqa --batch_size 2
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
from peft import PeftModel
import time
import wandb


def process_batch_element(args):
    """
    Worker function that preprocesses a single sample in parallel.

    Takes: (image, question, index, ground_truth, max_dimension, cot_instruction)
    Returns: dict with preprocessed data, or None if image is missing

    Does image resizing + message dict creation.
    Note: processor.apply_chat_template() stays in main process (can't pickle processor).
    """
    image, question, idx, ground_truth, max_dimension, cot_instruction = args

    # Skip samples without images
    if image is None:
        return None

    # Resize image if too large
    if image.width > max_dimension or image.height > max_dimension:
        image = image.copy()
        image.thumbnail((max_dimension, max_dimension))

    if cot_instruction:
        question_with_instruction = f"{question}\n\n{cot_instruction}"
    else:
        question_with_instruction = question

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question_with_instruction}
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
parser.add_argument('--adapter_artifact', type=str, default=None, help='WandB artifact name for LoRA adapter (e.g., "my-model:v0" or "entity/project/my-model:v0")')
parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16', 'float32'])

# Dataset args
parser.add_argument('--dataset', type=str, default='mathverse', choices=['mathverse', 'scienceqa'])
parser.add_argument('--num_samples', type=int, default=None, help='Number of samples (None=all)')

# Generation args
parser.add_argument('--max_new_tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.0)
parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference')
parser.add_argument('--cot_instruction', type=str, default=None,
                    help='Chain-of-thought instruction to append to questions (e.g., "Think step by step and explain your reasoning before selecting your final answer.")')

# Output
parser.add_argument('--output_dir', type=str, default='experiments/baseline/predictions')

# WandB args
parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='stem-vlm', help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='WandB tags')

args = parser.parse_args()

# Load config file if provided
if args.config:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with config values (command-line args take priority)
    if 'model' in config:
        args.model_name = config['model'].get('name', args.model_name)
        args.adapter_artifact = config['model'].get('adapter_artifact', args.adapter_artifact)
        args.dtype = config['model'].get('dtype', args.dtype)
    if 'dataset' in config:
        args.dataset = config['dataset'].get('name', args.dataset)
        args.num_samples = config['dataset'].get('num_samples', args.num_samples)
    if 'generation' in config:
        args.max_new_tokens = config['generation'].get('max_new_tokens', args.max_new_tokens)
        args.temperature = config['generation'].get('temperature', args.temperature)
        args.batch_size = config['generation'].get('batch_size', args.batch_size)
        args.cot_instruction = config['generation'].get('cot_instruction', args.cot_instruction)
    if 'output' in config:
        args.output_dir = config['output'].get('dir', args.output_dir)
    if 'wandb' in config:
        args.use_wandb = config['wandb'].get('enabled', args.use_wandb)
        args.wandb_project = config['wandb'].get('project', args.wandb_project)
        args.wandb_entity = config['wandb'].get('entity', args.wandb_entity)
        args.wandb_run_name = config['wandb'].get('run_name', args.wandb_run_name)
        args.wandb_tags = config['wandb'].get('tags', args.wandb_tags)

print("=" * 70)
print("STEM-VLM Inference")
print("=" * 70)
print(f"Model: {args.model_name}")
if args.adapter_artifact:
    print(f"Adapter Artifact: {args.adapter_artifact}")
print(f"Dataset: {args.dataset}")
print(f"Num samples: {args.num_samples if args.num_samples else 'all'}")
print(f"Batch size: {args.batch_size}")
print(f"Output: {args.output_dir}")
print(f"WandB: {'enabled' if args.use_wandb else 'disabled'}")
print("=" * 70)

# Download adapter from WandB if artifact is specified
adapter_path = None
if args.adapter_artifact:
    print(f"\nDownloading adapter from WandB artifact: {args.adapter_artifact}")
    api = wandb.Api()

    artifact_parts = args.adapter_artifact.split('/')

    if len(artifact_parts) == 3:
        # Full path provided
        artifact_ref = args.adapter_artifact
    elif len(artifact_parts) == 2:
        # Project/artifact provided
        if args.wandb_entity:
            artifact_ref = f"{args.wandb_entity}/{args.adapter_artifact}"
        else:
            # Use default entity from wandb login
            artifact_ref = args.adapter_artifact
    else:
        # Just artifact name
        if args.wandb_entity:
            artifact_ref = f"{args.wandb_entity}/{args.wandb_project}/{args.adapter_artifact}"
        else:
            # Let WandB resolve using default entity
            artifact_ref = f"{args.wandb_project}/{args.adapter_artifact}"

    print(f"Resolving artifact: {artifact_ref}")
    artifact = api.artifact(artifact_ref, type="model")
    adapter_path = artifact.download()
    print(f"✓ Adapter downloaded to: {adapter_path}")

# Initialize WandB if enabled
wandb_run = None
if args.use_wandb:
    # Auto-generate run name if not provided
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.wandb_run_name is None:
        args.wandb_run_name = f"inference-{args.dataset}-{timestamp}"

    # Prepare tags
    tags = args.wandb_tags if args.wandb_tags else []
    tags.append("inference")
    tags.append(args.dataset)
    if args.adapter_artifact:
        tags.append("finetuned")
    else:
        tags.append("baseline")

    # Initialize WandB run
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config=vars(args),
        tags=tags,
        notes=f"Inference on {args.dataset} using {args.model_name}"
    )
    print(f"✓ WandB run initialized: {wandb_run.url}")

    # Link model artifact to this run for lineage tracking
    if args.adapter_artifact:
        if args.wandb_entity:
            artifact_ref = f"{args.wandb_entity}/{args.wandb_project}/{args.adapter_artifact}"
        else:
            artifact_ref = f"{args.wandb_project}/{args.adapter_artifact}"
        wandb_run.use_artifact(artifact_ref, type="model")
        print(f"✓ Linked adapter artifact to run")



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
model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_name,
    torch_dtype=dtype,
    device_map="auto",
    trust_remote_code=True
)

# Load LoRA adapter if provided
if adapter_path:
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)

model.eval()

print(f"Model loaded on {model.device}")
if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory: {mem_gb:.2f} GB")


if args.dataset == 'mathverse':
    dataset = load_dataset("AI4Math/MathVerse", "testmini")['testmini']
elif args.dataset == 'scienceqa':
    dataset = load_dataset("derek-thomas/ScienceQA", split="test")

if args.num_samples:
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))

print(f"✓ Loaded {len(dataset)} samples")



results = []

# Track inference timing
inference_start_time = time.time()

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
            MAX_DIMENSION,
            args.cot_instruction
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

# Calculate inference time
inference_time = time.time() - inference_start_time

# Save predictions to JSON
predictions_file = output_dir / f"{args.dataset}_predictions_{timestamp}.json"
predictions_data = {
    "metadata": {
        "model": args.model_name,
        "adapter_artifact": args.adapter_artifact,
        "adapter_path": adapter_path,
        "dataset": args.dataset,
        "num_samples": len(results),
        "timestamp": timestamp,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "batch_size": args.batch_size,
        "cot_instruction": args.cot_instruction,
        "uses_cot": args.cot_instruction is not None,
        "inference_time_seconds": inference_time,
        "wandb_run_id": wandb_run.id if wandb_run else None,
    },
    "predictions": results
}

with open(predictions_file, 'w') as f:
    json.dump(predictions_data, f, indent=2)

print(f"✓ Predictions saved to: {predictions_file}")

# Save config used for this run
config_file = output_dir / f"config_{timestamp}.yaml"
with open(config_file, 'w') as f:
    yaml.dump(vars(args), f)
print(f"✓ Config saved to: {config_file}")

# Log to WandB if enabled
if wandb_run:
    # Log summary metrics
    wandb.log({
        "num_samples": len(results),
        "inference_time_seconds": inference_time,
        "avg_time_per_sample": inference_time / len(results) if len(results) > 0 else 0,
    })

    # Create predictions table for first few samples (for inspection in WandB UI)
    sample_size = min(20, len(results))  # Log first 20 samples as table
    table = wandb.Table(columns=["sample_id", "question", "prediction", "ground_truth"])
    for result in results[:sample_size]:
        table.add_data(
            result["sample_id"],
            result["question"][:100] + "..." if len(result["question"]) > 100 else result["question"],
            result["prediction"],
            str(result["ground_truth"])
        )
    wandb.log({"predictions_sample": table})

    # Save predictions file as WandB artifact
    predictions_artifact = wandb.Artifact(
        name=f"{args.dataset}-predictions",
        type="predictions",
        description=f"Predictions on {args.dataset} using {args.model_name}",
        metadata={
            "model": args.model_name,
            "dataset": args.dataset,
            "num_samples": len(results),
            "inference_time": inference_time,
        }
    )
    predictions_artifact.add_file(str(predictions_file))
    wandb_run.log_artifact(predictions_artifact)

    print(f"✓ Results logged to WandB: {wandb_run.url}")

    # Finish WandB run
    wandb_run.finish()

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
