"""
LLM-as-Judge evaluation system using GPT-4o-mini.

Provides:
- LLMJudge: API client with rate limiting and retry logic
- Helper functions: prompt formatting, response parsing, metrics calculation
"""

import time
from typing import Dict, List
from openai import OpenAI
from openai import RateLimitError, APIError


class LLMJudge:
    """
    GPT-4o-mini judge client with rate limiting and automatic retries.

    Handles all communication with the OpenAI API, including:
    - Rate limiting (automatic delays between requests)
    - Retry logic with exponential backoff
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: int = 200,
        requests_per_minute: int = 100,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        """
        Initialize the LLM judge client.

        Args:
            model: Model name (e.g., "gpt-4o-mini")
            api_key: OpenAI API key
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
            max_tokens: Maximum tokens in response
            requests_per_minute: Max requests per minute (for rate limiting)
            max_retries: How many times to retry failed requests
            retry_delay: Initial delay (seconds) before first retry (doubles each time)
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self.client = OpenAI(api_key=api_key)

        # Calculate delay between requests to respect rate limits
        self.request_delay = 60.0 / requests_per_minute

    def judge_single(self, system_prompt: str, user_prompt: str) -> str:
        """
        Send a single judgment request to GPT-4o-mini.

        Includes retry logic with exponential backoff for handling transient errors.

        Args:
            system_prompt: System message (instructions for the judge)
            user_prompt: User message (the actual evaluation task)

        Returns:
            str: The judge's response text

        Raises:
            Exception: If all retries fail or a permanent error occurs
        """
        for attempt in range(self.max_retries):
            try:
                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )

                # Extract and return the text response
                return response.choices[0].message.content

            except RateLimitError as e:
                # Rate limit hit - wait longer before retrying
                if attempt == self.max_retries - 1:
                    raise  # Give up after max retries

                wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                print(f"\nRate limit hit. Waiting {wait_time}s before retry (attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(wait_time)

            except APIError as e:
                # Server error (500, 503, etc.) - transient, worth retrying
                if attempt == self.max_retries - 1:
                    raise

                wait_time = self.retry_delay * (2 ** attempt)
                print(f"\nAPI error: {e}. Retrying in {wait_time}s (attempt {attempt + 1}/{self.max_retries})...")
                time.sleep(wait_time)

            except Exception as e:
                # Unknown error - don't retry, just fail
                print(f"\nFatal error: {e}")
                raise

        raise RuntimeError("Max retries exceeded")

    def wait_for_rate_limit(self):
        """
        Sleep to respect rate limits.

        Call this after each judge_single() call to add the required delay.
        """
        time.sleep(self.request_delay)


def format_judge_prompt(template: str, question: str, ground_truth: str, prediction: str) -> str:
    """
    Fill in the prompt template with sample data.

    Takes a template string with {placeholders} and replaces them with actual values.

    Args:
        template: Prompt template with {question}, {ground_truth}, {prediction} placeholders
        question: The question text
        ground_truth: The correct answer
        prediction: The model's predicted answer

    Returns:
        str: Formatted prompt ready to send to the judge
    """
    
    return template.format(
        question=question,
        ground_truth=ground_truth,
        prediction=prediction
    )


def parse_judge_response(response: str) -> Dict[str, any]:
    """
    Parse the judge's response into structured format.

    Expected response format:
        Verdict: CORRECT
        Reasoning: The answer is mathematically equivalent.

    Args:
        response: Raw text response from the judge

    Returns:
        dict: Contains:
            - correct (bool): True if verdict is CORRECT, False if INCORRECT
            - reasoning (str): Explanation from the judge
            - raw_response (str): Original response text
    """
    result = {
        'correct': None,
        'reasoning': None,
        'raw_response': response
    }

    # Split response into lines
    lines = response.strip().split('\n')

    for line in lines:
        # Look for "Verdict: CORRECT" or "Verdict: INCORRECT"
        if line.strip().startswith('Verdict:'):
            verdict_text = line.split(':', 1)[1].strip().upper()
            result['correct'] = ('CORRECT' in verdict_text)

        # Look for "Reasoning: ..."
        elif line.strip().startswith('Reasoning:'):
            result['reasoning'] = line.split(':', 1)[1].strip()

    return result


def calculate_accuracy(results: List[Dict]) -> Dict[str, float]:
    """
    Calculate accuracy metrics from judge results.

    Args:
        results: List of judgment dicts, each with a 'correct' field

    Returns:
        dict: Contains:
            - accuracy (float): Percentage correct (0.0 to 1.0)
            - total (int): Total number of samples
            - correct (int): Number of correct predictions
            - incorrect (int): Number of incorrect predictions
    """
    total = len(results)
    correct = sum(1 for r in results if r.get('correct') == True)
    incorrect = total - correct
    accuracy = correct / total if total > 0 else 0.0

    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'incorrect': incorrect
    }
