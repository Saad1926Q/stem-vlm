"""
Data utilities for ScienceQA dataset formatting.
"""

from datasets import load_dataset


def format_sample(sample):
    """
    Convert a single sample to Unsloth training format.

    Returns dict with:
        - messages: chat messages structure (for UnslothVisionDataCollator)
        - question: original question (for reference)
        - answer: original answer (for reference)

    Returns None if sample is missing required fields.
    """
    image = sample.get('image')
    question = sample.get('question', '')
    answer = sample.get('answer', '')

    # Skip if missing required fields
    if image is None or not question or not answer:
        return None

    # Create messages structure that UnslothVisionDataCollator expects
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": str(answer)}
            ]
        }
    ]

    return {
        "messages": messages,
        "question": question,
        "answer": str(answer)
    }


def load_and_format_scienceqa(split='train', max_samples=None):
    """
    Load ScienceQA dataset and format samples.

    Args:
        split: Dataset split ('train' or 'validation')
        max_samples: Maximum number of samples to process (None = all)

    Returns:
        List of formatted samples
    """
    print(f"Loading ScienceQA {split} split...")
    dataset = load_dataset("derek-thomas/ScienceQA")
    data = dataset[split]

    if max_samples and len(data) > max_samples:
        data = data.select(range(max_samples))
        print(f"Limited to {max_samples} samples")

    print(f"Formatting {len(data)} samples...")
    formatted = [format_sample(sample) for sample in data]

    # Filter out None values (samples with missing data)
    formatted = [s for s in formatted if s is not None]

    print(f"Processed {len(formatted)} samples (skipped {len(data) - len(formatted)} with missing data)")

    return formatted
