"""
Train Qwen2-VL model on ScienceQA using Unsloth.

Usage:
    python scripts/train.py --config configs/train_scienceqa.yaml
    python scripts/train.py --num_epochs 3 --max_samples 1000
    python scripts/train.py --config configs/train.yaml --learning_rate 5e-5
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import argparse
import yaml
import torch
from datetime import datetime
import subprocess
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
import wandb
from data.data_utils import load_and_format_scienceqa


parser = argparse.ArgumentParser(description="Train Qwen2-VL on ScienceQA with Unsloth")

# Config file
parser.add_argument('--config', type=str, help='Path to YAML config file')

# Data args
parser.add_argument('--max_samples', type=int, default=None,
                    help='Maximum number of samples to use for training/validation (None = all)')

# Model args
parser.add_argument('--model_name', type=str, default='Qwen/Qwen2-VL-2B-Instruct')
parser.add_argument('--load_in_4bit', action='store_true', default=True)
parser.add_argument('--max_seq_length', type=int, default=2048)

# LoRA args
parser.add_argument('--lora_r', type=int, default=16)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)

# Training hyperparameters
parser.add_argument('--num_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--warmup_ratio', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--max_grad_norm', type=float, default=1.0)
parser.add_argument('--lr_scheduler_type', type=str, default='linear')

# Checkpointing
parser.add_argument('--save_steps', type=int, default=100)
parser.add_argument('--save_strategy', type=str, default='steps', choices=['steps', 'epoch', 'no'])
parser.add_argument('--save_total_limit', type=int, default=3)
parser.add_argument('--load_best_at_end', action='store_true', default=True)
parser.add_argument('--resume_from_checkpoint', type=str, default=None)

# WandB
parser.add_argument('--use_wandb', action='store_true', default=True)
parser.add_argument('--wandb_project', type=str, default='stem-vlm')
parser.add_argument('--wandb_entity', type=str, default=None)
parser.add_argument('--wandb_run_name', type=str, default=None)

# Experiment / Model Registry
parser.add_argument('--experiment_name', type=str, default='baseline',
                    help='Name of experiment for grouping models (e.g., "baseline", "lora-rank-sweep", "lr-sweep")')
parser.add_argument('--experiment_tags', type=str, nargs='*', default=[],
                    help='Additional custom tags for the model (e.g., "production-candidate", "ablation-study")')

# Output
parser.add_argument('--output_dir', type=str, default='experiments/runs')
parser.add_argument('--run_name', type=str, default=None)

args = parser.parse_args()


if args.config:
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if 'data' in config:
        args.max_samples = config['data'].get('max_samples', args.max_samples)

    if 'model' in config:
        args.model_name = config['model'].get('name', args.model_name)
        args.load_in_4bit = config['model'].get('load_in_4bit', args.load_in_4bit)
        args.max_seq_length = config['model'].get('max_seq_length', args.max_seq_length)

    if 'lora' in config:
        args.lora_r = config['lora'].get('r', args.lora_r)
        args.lora_alpha = config['lora'].get('alpha', args.lora_alpha)
        args.lora_dropout = config['lora'].get('dropout', args.lora_dropout)

    if 'training' in config:
        args.num_epochs = config['training'].get('num_epochs', args.num_epochs)
        args.batch_size = config['training'].get('batch_size', args.batch_size)
        args.gradient_accumulation_steps = config['training'].get('gradient_accumulation_steps', args.gradient_accumulation_steps)
        args.learning_rate = config['training'].get('learning_rate', args.learning_rate)
        args.warmup_ratio = config['training'].get('warmup_ratio', args.warmup_ratio)
        args.weight_decay = config['training'].get('weight_decay', args.weight_decay)
        args.max_grad_norm = config['training'].get('max_grad_norm', args.max_grad_norm)
        args.lr_scheduler_type = config['training'].get('lr_scheduler_type', args.lr_scheduler_type)

    if 'checkpointing' in config:
        args.save_steps = config['checkpointing'].get('save_steps', args.save_steps)
        args.save_strategy = config['checkpointing'].get('save_strategy', args.save_strategy)
        args.save_total_limit = config['checkpointing'].get('save_total_limit', args.save_total_limit)
        args.load_best_at_end = config['checkpointing'].get('load_best_at_end', args.load_best_at_end)

    if 'wandb' in config:
        args.use_wandb = config['wandb'].get('enabled', args.use_wandb)
        args.wandb_project = config['wandb'].get('project', args.wandb_project)
        args.wandb_entity = config['wandb'].get('entity', args.wandb_entity)
        args.wandb_run_name = config['wandb'].get('run_name', args.wandb_run_name)

    if 'experiment' in config:
        args.experiment_name = config['experiment'].get('name', args.experiment_name)
        args.experiment_tags = config['experiment'].get('tags', args.experiment_tags)

    if 'output' in config:
        args.output_dir = config['output'].get('dir', args.output_dir)
        args.run_name = config['output'].get('run_name', args.run_name)


if args.run_name is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_name.split('/')[-1].lower()
    args.run_name = f"{model_short}-scienceqa-{timestamp}"

output_path = Path(args.output_dir) / args.run_name
output_path.mkdir(parents=True, exist_ok=True)

if args.use_wandb:
    wandb_run_name = args.wandb_run_name if args.wandb_run_name else args.run_name
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=wandb_run_name,
        config=vars(args)
    )

print("=" * 70)
print("STEM-VLM Fine Tuning with Unsloth")
print("=" * 70)
print(f"Model: {args.model_name}")
print(f"Run name: {args.run_name}")
print(f"Output: {output_path}")
print(f"LoRA rank: {args.lora_r}")
print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
print(f"Learning rate: {args.learning_rate}")
print(f"Epochs: {args.num_epochs}")
print(f"WandB: {'Enabled' if args.use_wandb else 'Disabled'}")
print("=" * 70)


print("\nLoading and formatting training data...")
train_dataset = load_and_format_scienceqa(split='train', max_samples=args.max_samples)

print("\nLoading and formatting validation data...")
val_dataset = load_and_format_scienceqa(split='validation', max_samples=args.max_samples)



model, tokenizer = FastVisionModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=args.max_seq_length,
    load_in_4bit=args.load_in_4bit,
)

print("Model loaded")
if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory: {mem_gb:.2f} GB")

model = FastVisionModel.get_peft_model(
    model,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print("LoRA adapters added")
print(f"Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

FastVisionModel.for_training(model)




training_args = SFTConfig(
    output_dir=str(output_path),
    num_train_epochs=args.num_epochs,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    lr_scheduler_type=args.lr_scheduler_type,

    logging_steps=10,
    logging_dir=str(output_path / "logs"),

    save_strategy=args.save_strategy,
    save_steps=args.save_steps if args.save_strategy == 'steps' else None,
    save_total_limit=args.save_total_limit,
    load_best_model_at_end=args.load_best_at_end if val_dataset else False,

    eval_strategy="steps" if val_dataset else "no",
    eval_steps=args.save_steps if val_dataset else None,

    report_to="wandb" if args.use_wandb else "none",

    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",

    seed=42,

    # Vision model specific settings
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    max_seq_length=args.max_seq_length,
)


print("\nInitializing trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
)

print("Trainer initialized")

config_save_path = output_path / "config.yaml"
with open(config_save_path, 'w') as f:
    yaml.dump(vars(args), f, default_flow_style=False)
print(f"Config saved to: {config_save_path}")



print("\n" + "=" * 70)
print("Starting training...")
print("=" * 70 + "\n")

resume_checkpoint = args.resume_from_checkpoint
if resume_checkpoint == "True":
    resume_checkpoint = True

trainer.train(resume_from_checkpoint=resume_checkpoint)

print("\n" + "=" * 70)
print("Training complete!")
print("=" * 70)


def extract_model_name(model_path):
    """Extract model name from path: Qwen/Qwen2-VL-2B-Instruct -> Qwen2-VL-2B-Instruct"""
    return model_path.split('/')[-1]


final_model_path = output_path / "checkpoint-final"
print(f"\nSaving final model to: {final_model_path}")
model.save_pretrained(str(final_model_path))
tokenizer.save_pretrained(str(final_model_path))
print("Final model saved")

# Model Registry: Save to WandB artifacts
if args.use_wandb:
    
    model_name = extract_model_name(args.model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"{model_name}-{args.experiment_name}-{timestamp}"

    final_metrics = {}
    if trainer.state.log_history:
        train_losses = [log['loss'] for log in trainer.state.log_history if 'loss' in log]
        if train_losses:
            final_metrics['final_train_loss'] = train_losses[-1]

        eval_losses = [log['eval_loss'] for log in trainer.state.log_history if 'eval_loss' in log]
        if eval_losses:
            final_metrics['final_eval_loss'] = eval_losses[-1]
            final_metrics['best_eval_loss'] = min(eval_losses)

    metadata = {
        "model_name": model_name,

        # LoRA hyperparameters
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,

        # Training hyperparameters
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "num_epochs": args.num_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,

        # Dataset info
        "max_samples": args.max_samples if args.max_samples else "all",
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset) if val_dataset else 0,

        # Performance metrics
        **final_metrics,

        # Experiment info
        "experiment_name": args.experiment_name,
        "run_name": args.run_name,
        "training_date": datetime.now().isoformat(),
    }

    # Build tags
    tags = [model_name, args.experiment_name]
    tags.extend(args.experiment_tags)

    # Add dataset info tag
    if args.max_samples:
        tags.append(f"samples-{args.max_samples}")
    else:
        tags.append("full-dataset")

    print(f"\nArtifact name: {artifact_name}")
    print(f"Tags: {', '.join(tags)}")

    # Create and log artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"Fine-tuned {model_name} on ScienceQA ({args.experiment_name})",
        metadata=metadata
    )

    artifact.add_dir(str(final_model_path))
    wandb.log_artifact(artifact, aliases=tags)

    print("\nModel saved to WandB Model Registry!")
    print(f"  View at: {wandb.run.url}")

    wandb.finish()

print("\n" + "=" * 70)
print("All done!")
print("=" * 70)
print(f"\nCheckpoints saved to: {output_path}")
print(f"Final model: {final_model_path}")