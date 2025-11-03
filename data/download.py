"""
Simple script to download datasets from HuggingFace.
Run this first to cache datasets locally.
"""

from datasets import load_dataset

print("=" * 60)
print("Downloading MathVista...")
print("=" * 60)
mathvista = load_dataset("AI4Math/MathVista", split="test")

print(f"Downloaded {len(mathvista)} samples")

print("\n" + "=" * 60)
print("Downloading ScienceQA...")
print("=" * 60)
scienceqa = load_dataset("derek-thomas/ScienceQA", split="test")
print(f"✓ Downloaded {len(scienceqa)} samples")