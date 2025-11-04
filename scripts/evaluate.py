"""
Evaluate model predictions using LLM-as-Judge (GPT-4o-mini).

Usage:
    export OPENAI_API_KEY="sk-..."
    python scripts/evaluate.py \
        --predictions experiments/baseline/mathverse_predictions_20250104.json \
        --config configs/judge.yaml
"""

import argparse
import yaml
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

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

print(f"Judge: {config['judge']['provider']}/{config['judge']['model']}")
print(f"Predictions: {args.predictions}")
print("=" * 70)


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
print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
