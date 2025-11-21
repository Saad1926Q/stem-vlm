"""
Data utilities for ScienceQA dataset formatting.
"""

from datasets import load_dataset


def format_sample(sample, use_cot=False):
    """
    Convert a single sample to Unsloth training format.

    Args:
        sample: Dataset sample
        use_cot: If True, include lecture + solution in assistant response

    Returns dict with:
        - messages: chat messages structure (for UnslothVisionDataCollator)
        - question: original question (for reference)
        - answer: original answer (for reference)

    Returns None if sample is missing required fields.
    """
    image = sample.get('image')
    question = sample.get('question', '')
    answer = sample.get('answer', '')
    lecture = sample.get('lecture', '')
    solution = sample.get('solution', '')

    # Skip if missing required fields
    if image is None or not question or not answer:
        return None

    # Build assistant response
    if use_cot:
        parts = ["Let me work through this step by step."]
        if lecture and str(lecture).strip():
            parts.append(f"\nThe key concept here is: {lecture}")
        if solution and str(solution).strip():
            parts.append(f"\nNow, let's solve this:\n{solution}")
        parts.append(f"\nTherefore, the answer is: {answer}")
        assistant_text = "\n".join(parts)
    else:
        assistant_text = str(answer)

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
                {"type": "text", "text": assistant_text}
            ]
        }
    ]

    return {
        "messages": messages,
        "question": question,
        "answer": str(answer)
    }


def load_and_format_scienceqa(split='train', max_samples=None, use_cot=False):
    """
    Load ScienceQA dataset and format samples.

    Args:
        split: Dataset split ('train' or 'validation')
        max_samples: Maximum number of samples to process (None = all)
        use_cot: If True, include lecture + solution in assistant response

    Returns:
        List of formatted samples
    """
    print(f"Loading ScienceQA {split} split...")
    dataset = load_dataset("derek-thomas/ScienceQA")
    data = dataset[split]

    if max_samples and len(data) > max_samples:
        data = data.select(range(max_samples))
        print(f"Limited to {max_samples} samples")

    print(f"Formatting {len(data)} samples (use_cot={use_cot})...")
    formatted = [format_sample(sample, use_cot=use_cot) for sample in data]

    # Filter out None values (samples with missing data)
    formatted = [s for s in formatted if s is not None]

    print(f"Processed {len(formatted)} samples (skipped {len(data) - len(formatted)} with missing data)")

    return formatted
