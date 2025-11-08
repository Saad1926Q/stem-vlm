"""
Evaluate model predictions using LLM-as-Judge (GPT-4o-mini).

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/evaluate.py \
        --predictions experiments/baseline/mathverse_predictions_20250104.json \
        --config configs/judge.yaml
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import json
import os
from datetime import datetime
from tqdm import tqdm
import wandb

from evaluation.judge import (
    LLMJudge,
    format_judge_prompt,
    parse_judge_response,
    calculate_accuracy
)


parser = argparse.ArgumentParser(description="Evaluate predictions with LLM judge")

# Input/Output
parser.add_argument('--predictions', type=str, required=True, help='Path to predictions JSON file')
parser.add_argument('--config', type=str, default='configs/judge.yaml', help='Path to judge config')
parser.add_argument('--output_dir', type=str, help='Override output directory from config')

# Judge settings (optional overrides)
parser.add_argument('--judge_model', type=str, help='Override judge model')
parser.add_argument('--temperature', type=float, help='Override temperature')

# WandB args
parser.add_argument('--use_wandb', action='store_true', help='Enable WandB logging')
parser.add_argument('--wandb_project', type=str, default='stem-vlm', help='WandB project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity/username')
parser.add_argument('--wandb_run_name', type=str, default=None, help='WandB run name')
parser.add_argument('--wandb_tags', type=str, nargs='+', default=None, help='WandB tags')

# Model artifact (to update with accuracy)
parser.add_argument('--model_artifact', type=str, default=None,
                    help='Model artifact name to update with accuracy (e.g., "Qwen2-VL-2B-Instruct-baseline-20250107_123456:v0")')

args = parser.parse_args()


# Load configuration
print("=" * 70)
print("LLM-as-Judge Evaluation")
print("=" * 70)

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Override config with command-line args if provided
if args.judge_model:
    config['judge']['model'] = args.judge_model
if args.temperature is not None:
    config['judge']['temperature'] = args.temperature
if args.output_dir:
    config['output']['save_dir'] = args.output_dir
if 'wandb' in config:
    args.use_wandb = config['wandb'].get('enabled', args.use_wandb)
    args.wandb_project = config['wandb'].get('project', args.wandb_project)
    args.wandb_entity = config['wandb'].get('entity', args.wandb_entity)
    args.wandb_run_name = config['wandb'].get('run_name', args.wandb_run_name)
    args.wandb_tags = config['wandb'].get('tags', args.wandb_tags)

print(f"Judge: {config['judge']['provider']}/{config['judge']['model']}")
print(f"Predictions: {args.predictions}")
print(f"WandB: {'enabled' if args.use_wandb else 'disabled'}")
print("=" * 70)

# Initialize WandB if enabled
wandb_run = None
if args.use_wandb:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.wandb_run_name is None:
        predictions_basename = Path(args.predictions).stem
        args.wandb_run_name = f"eval-{predictions_basename}-{timestamp}"

    tags = args.wandb_tags if args.wandb_tags else []
    tags.append("evaluation")

    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        config={
            "judge_model": config['judge']['model'],
            "judge_temperature": config['judge']['temperature'],
            "predictions_file": args.predictions,
        },
        tags=tags,
        notes=f"Evaluation using {config['judge']['model']}"
    )
    print(f"✓ WandB run initialized: {wandb_run.url}")


# Load API key from environment
api_key_env = config['judge']['api_key_env']
api_key = os.getenv(api_key_env)

if not api_key:
    raise ValueError(
        f"API key not found in environment variable: {api_key_env}\n"
        f"Please set it with: export {api_key_env}='your-key-here'"
    )


# Initialize judge
judge = LLMJudge(
    model=config['judge']['model'],
    api_key=api_key,
    temperature=config['judge']['temperature'],
    max_tokens=config['judge']['max_tokens'],
    requests_per_minute=config['judge']['requests_per_minute'],
    max_retries=config['judge']['max_retries'],
    retry_delay=config['judge']['retry_delay_seconds']
)

print(f"✓ Judge initialized: {config['judge']['model']}")
print(f"✓ Rate limit: {config['judge']['requests_per_minute']} requests/min")


# Load predictions
with open(args.predictions, 'r') as f:
    data = json.load(f)

predictions = data['predictions']
metadata = data['metadata']

print(f"✓ Loaded {len(predictions)} predictions from {metadata.get('dataset', 'unknown')} dataset")
print(f"✓ Base model: {metadata.get('model', 'unknown')}")
print("=" * 70)


# Evaluate each prediction
results = []
system_prompt = config['prompts']['system']
user_template = config['prompts']['user_template']

print("Starting evaluation...")

for i, pred in enumerate(tqdm(predictions, desc="Judging", unit="sample")):
    # Format the evaluation prompt
    user_prompt = format_judge_prompt(
        template=user_template,
        question=pred['question'],
        ground_truth=pred['ground_truth'],
        prediction=pred['prediction']
    )

    # Get judgment from GPT-4o-mini
    try:
        response = judge.judge_single(system_prompt, user_prompt)

        # Parse the response
        judgment = parse_judge_response(response)

        # Store result
        results.append({
            'sample_id': pred['sample_id'],
            'question': pred['question'],
            'ground_truth': pred['ground_truth'],
            'prediction': pred['prediction'],
            'correct': judgment['correct'],
            'reasoning': judgment['reasoning'],
            'raw_judge_response': judgment['raw_response'] if config['output']['save_raw_responses'] else None
        })

    except Exception as e:
        # If judgment fails, record the error
        print(f"\n[ERROR] Failed to judge sample {pred['sample_id']}: {e}")
        results.append({
            'sample_id': pred['sample_id'],
            'question': pred['question'],
            'ground_truth': pred['ground_truth'],
            'prediction': pred['prediction'],
            'correct': None,
            'reasoning': f"Error: {str(e)}",
            'raw_judge_response': None
        })

    # Rate limiting: wait before next request (except for last sample)
    if i < len(predictions) - 1:
        judge.wait_for_rate_limit()


# Calculate accuracy
metrics = calculate_accuracy(results)

print("\n" + "=" * 70)
print("Results")
print("=" * 70)
print(f"Total samples: {metrics['total']}")
print(f"Correct: {metrics['correct']}")
print(f"Incorrect: {metrics['incorrect']}")
print(f"Accuracy: {metrics['accuracy']:.2%}")
print("=" * 70)


# Save results
output_dir = Path(config['output']['save_dir'])
output_dir.mkdir(parents=True, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
dataset_name = metadata.get('dataset', 'unknown')
results_file = output_dir / f"{dataset_name}_judge_results_{timestamp}.json"

with open(results_file, 'w') as f:
    json.dump({
        'metadata': {
            'judge_model': config['judge']['model'],
            'judge_provider': config['judge']['provider'],
            'judge_temperature': config['judge']['temperature'],
            'base_model': metadata.get('model'),
            'dataset': metadata.get('dataset'),
            'timestamp': timestamp,
            'accuracy': metrics['accuracy'],
            'total_samples': metrics['total'],
            'correct_samples': metrics['correct'],
            'incorrect_samples': metrics['incorrect']
        },
        'results': results
    }, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")

# Log to WandB if enabled
if wandb_run:
    # Log summary metrics
    wandb.log({
        "accuracy": metrics['accuracy'],
        "total_samples": metrics['total'],
        "correct_samples": metrics['correct'],
        "incorrect_samples": metrics['incorrect']
    })

    # Create detailed results table
    sample_size = min(20, len(results))
    table = wandb.Table(columns=["sample_id", "question", "prediction", "ground_truth", "correct", "reasoning"])
    for result in results[:sample_size]:
        table.add_data(
            result["sample_id"],
            result["question"][:100] + "..." if len(result["question"]) > 100 else result["question"],
            result["prediction"][:100] + "..." if len(result["prediction"]) > 100 else result["prediction"],
            str(result["ground_truth"]),
            result["correct"],
            result["reasoning"][:200] + "..." if result["reasoning"] and len(result["reasoning"]) > 200 else result["reasoning"]
        )
    wandb.log({"evaluation_results_sample": table})

    # Save results file as WandB artifact
    results_artifact = wandb.Artifact(
        name=f"{dataset_name}-evaluation",
        type="evaluation",
        description=f"Evaluation results for {dataset_name}",
        metadata={
            "accuracy": metrics['accuracy'],
            "judge_model": config['judge']['model'],
            "total_samples": metrics['total'],
        }
    )
    results_artifact.add_file(str(results_file))
    wandb_run.log_artifact(results_artifact)

    # Try to link to inference run if wandb_run_id is in metadata
    if 'wandb_run_id' in metadata and metadata['wandb_run_id']:
        wandb_run.config.update({"inference_run_id": metadata['wandb_run_id']})

    # Update model artifact with accuracy if specified
    if args.model_artifact:
        print(f"\nUpdating model artifact with accuracy: {args.model_artifact}")
        api = wandb.Api()

        # Build full artifact reference
        if args.wandb_entity:
            artifact_ref = f"{args.wandb_entity}/{args.wandb_project}/{args.model_artifact}"
        else:
            artifact_ref = f"{args.wandb_project}/{args.model_artifact}"

        try:
            # Fetch the model artifact
            model_artifact = api.artifact(artifact_ref, type="model")

            # Update metadata with dataset-specific accuracy
            dataset_name = metadata.get('dataset', 'unknown')
            model_artifact.metadata[f'accuracy_{dataset_name}'] = metrics['accuracy']
            model_artifact.metadata[f'total_samples_{dataset_name}'] = metrics['total']
            model_artifact.metadata[f'correct_samples_{dataset_name}'] = metrics['correct']

            # Save the updated metadata
            model_artifact.save()

            print(f"✓ Model artifact updated with {dataset_name} accuracy: {metrics['accuracy']:.2%}")
        except Exception as e:
            print(f"Failed to update model artifact: {e}")

    print(f"✓ Results logged to WandB: {wandb_run.url}")
    wandb_run.finish()

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
