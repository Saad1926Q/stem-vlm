"""
LLM-as-Judge evaluation system using GPT-4o-mini.

Provides:
- LLMJudge: API client with rate limiting and retry logic
- Helper functions: prompt formatting, response parsing, metrics calculation
"""

import time
import base64
from io import BytesIO
from typing import Dict, List, Optional
from openai import OpenAI
from openai import RateLimitError, APIError
from PIL import Image


def image_to_base64(pil_image: Image.Image) -> str:
    """
    Convert a PIL Image to base64 string for OpenAI API.

    Args:
        pil_image: PIL Image object

    Returns:
        str: Base64 encoded image string
    """
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


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

    def judge_single(self, system_prompt: str, user_prompt: str, image: Optional[Image.Image] = None) -> str:
        """
        Send a single judgment request to GPT-4o-mini.

        Includes retry logic with exponential backoff for handling transient errors.
        Supports vision - if an image is provided, it will be sent along with the text.

        Args:
            system_prompt: System message (instructions for the judge)
            user_prompt: User message (the actual evaluation task)
            image: Optional PIL Image to send for visual evaluation

        Returns:
            str: The judge's response text

        Raises:
            Exception: If all retries fail or a permanent error occurs
        """
        for attempt in range(self.max_retries):
            try:
                # Build user message content
                if image is not None:
                    # Vision mode: send image + text
                    base64_image = image_to_base64(image)
                    user_content = [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": user_prompt
                        }
                    ]
                else:
                    # Text-only mode: just send text
                    user_content = user_prompt

                # Make the API call
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
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


def parse_judge_response(response: str, is_cot: bool = False) -> Dict[str, any]:
    """
    Parse the judge's response into structured format.

    Expected response formats:

    Standard (non-CoT):
        Verdict: CORRECT
        Reasoning: The answer is mathematically equivalent.

    CoT format:
        Answer: CORRECT
        Reasoning: INCORRECT
        Explanation: The final answer is correct but the calculation had errors.

    Args:
        response: Raw text response from the judge
        is_cot: Whether this is evaluating a CoT prediction (uses different format)

    Returns:
        dict: For standard evaluation:
            - correct (bool): True if verdict is CORRECT, False if INCORRECT
            - reasoning (str): Explanation from the judge
            - raw_response (str): Original response text

        For CoT evaluation:
            - answer_correct (bool): Whether final answer matches ground truth
            - reasoning_correct (bool): Whether reasoning is sound
            - explanation (str): Explanation from the judge
            - raw_response (str): Original response text
    """
    result = {
        'raw_response': response
    }

    # Split response into lines
    lines = response.strip().split('\n')

    if is_cot:
        # Parse CoT format: Answer, Reasoning, Explanation
        result['answer_correct'] = None
        result['reasoning_correct'] = None
        result['explanation'] = None

        for line in lines:
            line = line.strip()
            if line.startswith('Answer:'):
                verdict_text = line.split(':', 1)[1].strip().upper()
                result['answer_correct'] = ('INCORRECT' not in verdict_text)
            elif line.startswith('Reasoning:'):
                verdict_text = line.split(':', 1)[1].strip().upper()
                result['reasoning_correct'] = ('INCORRECT' not in verdict_text)
            elif line.startswith('Explanation:'):
                result['explanation'] = line.split(':', 1)[1].strip()
    else:
        # Parse standard format: Verdict, Reasoning
        result['correct'] = None
        result['reasoning'] = None

        for line in lines:
            line = line.strip()
            if line.startswith('Verdict:'):
                verdict_text = line.split(':', 1)[1].strip().upper()
                result['correct'] = ('INCORRECT' not in verdict_text)
            elif line.startswith('Reasoning:'):
                result['reasoning'] = line.split(':', 1)[1].strip()

    return result


def calculate_accuracy(results: List[Dict], is_cot: bool = False) -> Dict[str, float]:
    """
    Calculate accuracy metrics from judge results.

    Args:
        results: List of judgment dicts
        is_cot: Whether these are CoT evaluations (with separate answer/reasoning verdicts)

    Returns:
        dict: For standard evaluation:
            - accuracy (float): Percentage correct (0.0 to 1.0)
            - total (int): Total number of samples
            - correct (int): Number of correct predictions
            - incorrect (int): Number of incorrect predictions

        For CoT evaluation (additional metrics):
            - answer_accuracy (float): Percentage with correct final answer
            - reasoning_accuracy (float): Percentage with correct reasoning
            - both_correct (int): Both answer and reasoning correct
            - answer_correct_reasoning_wrong (int): Right answer, wrong reasoning
            - answer_wrong_reasoning_correct (int): Wrong answer, right reasoning
            - both_wrong (int): Both answer and reasoning wrong
    """
    total = len(results)

    if is_cot:
        # CoT metrics
        answer_correct_count = sum(1 for r in results if r.get('answer_correct') == True)
        reasoning_correct_count = sum(1 for r in results if r.get('reasoning_correct') == True)
        both_correct = sum(1 for r in results if r.get('answer_correct') == True and r.get('reasoning_correct') == True)
        answer_correct_reasoning_wrong = sum(1 for r in results if r.get('answer_correct') == True and r.get('reasoning_correct') == False)
        answer_wrong_reasoning_correct = sum(1 for r in results if r.get('answer_correct') == False and r.get('reasoning_correct') == True)
        both_wrong = sum(1 for r in results if r.get('answer_correct') == False and r.get('reasoning_correct') == False)

        return {
            'total': total,
            'answer_accuracy': answer_correct_count / total if total > 0 else 0.0,
            'reasoning_accuracy': reasoning_correct_count / total if total > 0 else 0.0,
            'both_correct': both_correct,
            'answer_correct_reasoning_wrong': answer_correct_reasoning_wrong,
            'answer_wrong_reasoning_correct': answer_wrong_reasoning_correct,
            'both_wrong': both_wrong,
        }
    else:
        # Standard metrics
        correct = sum(1 for r in results if r.get('correct') == True)
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0.0

        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct,
            'incorrect': incorrect
        }
