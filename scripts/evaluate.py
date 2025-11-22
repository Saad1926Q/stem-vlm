"""
Evaluate model predictions using LLM-as-Judge (GPT-4o-mini).

Usage Examples:

1. Basic evaluation (standard predictions):
   export OPENAI_API_KEY="sk-..."
   python scripts/evaluate.py \
       --predictions experiments/baseline/scienceqa_predictions_20250104.json \
       --config configs/judge.yaml

2. Evaluate Chain-of-Thought predictions (auto-detects from metadata):
   python scripts/evaluate.py \
       --predictions experiments/cot/scienceqa_predictions_20250104.json \
       --config configs/judge.yaml

3. With WandB logging:
   python scripts/evaluate.py \
       --predictions experiments/baseline/mathverse_predictions_20250104.json \
       --config configs/judge.yaml \
       --use_wandb \
       --wandb_project stem-vlm \
       --wandb_run_name "baseline-mathverse-eval" \
       --wandb_tags evaluation baseline mathverse

4. Update model artifact with accuracy metrics:
   python scripts/evaluate.py \
       --predictions experiments/baseline/scienceqa_predictions_20250104.json \
       --config configs/judge.yaml \
       --model_artifact "my-model:v0" \
       --use_wandb

5. Override judge settings:
   python scripts/evaluate.py \
       --predictions experiments/baseline/scienceqa_predictions_20250104.json \
       --config configs/judge.yaml \
       --judge_model "gpt-4o" \
       --temperature 0.0

6. Custom output directory:
   python scripts/evaluate.py \
       --predictions experiments/baseline/scienceqa_predictions_20250104.json \
       --config configs/judge.yaml \
       --output_dir experiments/baseline/evaluation_results

Notes:
- CoT evaluation is automatically enabled when predictions contain 'uses_cot: true' metadata
- For CoT predictions, the script evaluates both answer correctness AND reasoning quality
- Model artifacts are updated with dataset-specific accuracy (e.g., 'accuracy_scienceqa')
- CoT artifacts get both 'answer_accuracy_<dataset>' and 'reasoning_accuracy_<dataset>'
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
from dotenv import load_dotenv

from evaluation.judge import (
    LLMJudge,
    format_judge_prompt,
    parse_judge_response,
    calculate_accuracy
)
from datasets import load_dataset


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

parser.add_argument('--use_cot', action='store_true', default=None,
                    help='CoT evaluation mode')

args = parser.parse_args()

load_dotenv()

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
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError(
        f"API key not found in environment variable: OPENAI_API_KEY \n"
        f"Please set it with: export OPENAI_API_KEY=your-key-here"
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


if args.use_cot is not None:
    uses_cot = args.use_cot
else:
    uses_cot = metadata.get('uses_cot', False)

if uses_cot:
    system_prompt = config['prompts']['cot_system']
    user_template = config['prompts']['cot_user_template']

    dataset_name = metadata.get('dataset')

    if dataset_name == 'mathverse':
        dataset = load_dataset("AI4Math/MathVerse", "testmini")['testmini']
    elif dataset_name == 'scienceqa':
        dataset = load_dataset("derek-thomas/ScienceQA", split="test")
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create a mapping from sample_id to image for fast lookup
    image_map = {i: dataset[i]['image'] for i in range(len(dataset))}
else:
    system_prompt = config['prompts']['system']
    user_template = config['prompts']['user_template']
    image_map = None

results = []

print("Starting evaluation...")

for i, pred in enumerate(tqdm(predictions, desc="Judging", unit="sample")):
    # Format the evaluation prompt
    user_prompt = format_judge_prompt(
        template=user_template,
        question=pred['question'],
        ground_truth=pred['ground_truth'],
        prediction=pred['prediction']
    )

    # Get image if using CoT mode
    image = None
    if uses_cot and image_map is not None:
        sample_id = pred['sample_id']
        image = image_map.get(sample_id)

    # Get judgment from GPT-4o-mini
    try:
        response = judge.judge_single(system_prompt, user_prompt, image=image)

        # Parse the response
        judgment = parse_judge_response(response, is_cot=uses_cot)

        # Store result
        result = {
            'sample_id': pred['sample_id'],
            'question': pred['question'],
            'ground_truth': pred['ground_truth'],
            'prediction': pred['prediction'],
            'raw_judge_response': judgment['raw_response'] if config['output']['save_raw_responses'] else None
        }

        if uses_cot:
            result['answer_correct'] = judgment['answer_correct']
            result['reasoning_correct'] = judgment['reasoning_correct']
            result['explanation'] = judgment['explanation']
        else:
            result['correct'] = judgment['correct']
            result['reasoning'] = judgment['reasoning']

        results.append(result)

    except Exception as e:
        # If judgment fails, record the error
        print(f"\n[ERROR] Failed to judge sample {pred['sample_id']}: {e}")

        result = {
            'sample_id': pred['sample_id'],
            'question': pred['question'],
            'ground_truth': pred['ground_truth'],
            'prediction': pred['prediction'],
            'raw_judge_response': None
        }

        if uses_cot:
            result['answer_correct'] = None
            result['reasoning_correct'] = None
            result['explanation'] = f"Error: {str(e)}"
        else:
            result['correct'] = None
            result['reasoning'] = f"Error: {str(e)}"

        results.append(result)

    # Rate limiting: wait before next request (except for last sample)
    if i < len(predictions) - 1:
        judge.wait_for_rate_limit()


# Calculate accuracy
metrics = calculate_accuracy(results, is_cot=uses_cot)

print("\n" + "=" * 70)
print("Results")
print("=" * 70)
print(f"Total samples: {metrics['total']}")

if uses_cot:
    # CoT mode
    print(f"\nAnswer Accuracy: {metrics['answer_accuracy']:.2%}")
    print(f"Reasoning Accuracy: {metrics['reasoning_accuracy']:.2%}")
    print(f"\nBreakdown:")
    print(f"  Both Correct: {metrics['both_correct']} ({metrics['both_correct']/metrics['total']*100:.1f}%)")
    print(f"  Answer Correct, Reasoning Wrong: {metrics['answer_correct_reasoning_wrong']}")
    print(f"  Answer Wrong, Reasoning Correct: {metrics['answer_wrong_reasoning_correct']}")
    print(f"  Both Wrong: {metrics['both_wrong']}")
else:
    # Standard mode: show simple accuracy
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

results_metadata = {
    'judge_model': config['judge']['model'],
    'judge_provider': config['judge']['provider'],
    'judge_temperature': config['judge']['temperature'],
    'base_model': metadata.get('model'),
    'dataset': metadata.get('dataset'),
    'timestamp': timestamp,
    'uses_cot': uses_cot,
    'total_samples': metrics['total']
}

# Add appropriate metrics based on evaluation mode
if uses_cot:
    results_metadata.update({
        'answer_accuracy': metrics['answer_accuracy'],
        'reasoning_accuracy': metrics['reasoning_accuracy'],
        'both_correct': metrics['both_correct'],
        'answer_correct_reasoning_wrong': metrics['answer_correct_reasoning_wrong'],
        'answer_wrong_reasoning_correct': metrics['answer_wrong_reasoning_correct'],
        'both_wrong': metrics['both_wrong']
    })
else:
    results_metadata.update({
        'accuracy': metrics['accuracy'],
        'correct_samples': metrics['correct'],
        'incorrect_samples': metrics['incorrect']
    })

with open(results_file, 'w') as f:
    json.dump({
        'metadata': results_metadata,
        'results': results
    }, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")

# Log to WandB if enabled
if wandb_run:
    # Log summary metrics 
    if uses_cot:
        wandb.log({
            "answer_accuracy": metrics['answer_accuracy'],
            "reasoning_accuracy": metrics['reasoning_accuracy'],
            "total_samples": metrics['total'],
            "both_correct": metrics['both_correct'],
            "answer_correct_reasoning_wrong": metrics['answer_correct_reasoning_wrong'],
            "answer_wrong_reasoning_correct": metrics['answer_wrong_reasoning_correct'],
            "both_wrong": metrics['both_wrong']
        })
    else:
        wandb.log({
            "accuracy": metrics['accuracy'],
            "total_samples": metrics['total'],
            "correct_samples": metrics['correct'],
            "incorrect_samples": metrics['incorrect']
        })

    sample_size = min(20, len(results))

    if uses_cot:
        table = wandb.Table(columns=["sample_id", "question", "prediction", "ground_truth", "answer_correct", "reasoning_correct", "explanation"])
        for result in results[:sample_size]:
            table.add_data(
                result["sample_id"],
                result["question"][:100] + "..." if len(result["question"]) > 100 else result["question"],
                result["prediction"][:100] + "..." if len(result["prediction"]) > 100 else result["prediction"],
                str(result["ground_truth"]),
                result["answer_correct"],
                result["reasoning_correct"],
                result["explanation"][:200] + "..." if result["explanation"] and len(result["explanation"]) > 200 else result["explanation"]
            )
    else:
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
    artifact_metadata = {
        "judge_model": config['judge']['model'],
        "total_samples": metrics['total'],
        "uses_cot": uses_cot
    }

    if uses_cot:
        artifact_metadata.update({
            "answer_accuracy": metrics['answer_accuracy'],
            "reasoning_accuracy": metrics['reasoning_accuracy']
        })
    else:
        artifact_metadata["accuracy"] = metrics['accuracy']

    results_artifact = wandb.Artifact(
        name=f"{dataset_name}-evaluation",
        type="evaluation",
        description=f"Evaluation results for {dataset_name}",
        metadata=artifact_metadata
    )
    results_artifact.add_file(str(results_file))
    wandb_run.log_artifact(results_artifact)

    if 'wandb_run_id' in metadata and metadata['wandb_run_id']:
        wandb_run.config.update({"inference_run_id": metadata['wandb_run_id']})

    # Update model artifact with accuracy if specified
    if args.model_artifact:
        print(f"\nUpdating model artifact with metrics: {args.model_artifact}")
        api = wandb.Api()

        if args.wandb_entity:
            artifact_ref = f"{args.wandb_entity}/{args.wandb_project}/{args.model_artifact}"
        else:
            artifact_ref = f"{args.wandb_project}/{args.model_artifact}"

        try:
            # Fetch the model artifact
            model_artifact = api.artifact(artifact_ref, type="model")

            # Update metadata with dataset-specific metrics
            dataset_name = metadata.get('dataset', 'unknown')
            model_artifact.metadata[f'total_samples_{dataset_name}'] = metrics['total']

            if uses_cot:
                model_artifact.metadata[f'answer_accuracy_{dataset_name}'] = metrics['answer_accuracy']
                model_artifact.metadata[f'reasoning_accuracy_{dataset_name}'] = metrics['reasoning_accuracy']
                print(f"✓ Model artifact updated with {dataset_name} answer accuracy: {metrics['answer_accuracy']:.2%}, reasoning accuracy: {metrics['reasoning_accuracy']:.2%}")
            else:
                model_artifact.metadata[f'accuracy_{dataset_name}'] = metrics['accuracy']
                model_artifact.metadata[f'correct_samples_{dataset_name}'] = metrics['correct']
                print(f"✓ Model artifact updated with {dataset_name} accuracy: {metrics['accuracy']:.2%}")

            # Save the updated metadata
            model_artifact.save()

        except Exception as e:
            print(f"Failed to update model artifact: {e}")

    print(f"✓ Results logged to WandB: {wandb_run.url}")
    wandb_run.finish()

print("\n" + "=" * 70)
print("Done!")
print("=" * 70)
