"""
Simple script to download datasets from HuggingFace.
Run this first to cache datasets locally.
"""

from datasets import load_dataset

print("=" * 60)
print("Downloading MathVerse...")
print("=" * 60)
mathverse = load_dataset("AI4Math/MathVerse", "testmini")

print(f"Downloaded {len(mathverse['testmini'])} samples")

print("\n" + "=" * 60)
print("Downloading ScienceQA...")
print("=" * 60)
scienceqa = load_dataset("derek-thomas/ScienceQA", split="test")
print(f"✓ Downloaded {len(scienceqa)} samples")