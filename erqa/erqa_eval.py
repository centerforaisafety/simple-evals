#!/usr/bin/env python3
"""
ERQA (Embodied Reasoning QA) Evaluation Script
Evaluates vision-language models on embodied reasoning with multimodal questions.
"""

import asyncio
import os
import json
import re
import fire
import sys
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared import get_llm_agent_class, get_agent_config

# =============== PROMPTS ===============
SYSTEM_PROMPT = """You are evaluating embodied reasoning questions with multimodal content. You will be shown images and text that form a multiple-choice question. Please analyze the content carefully and provide your answer as a single letter (A, B, C, or D). Focus on spatial reasoning, object relationships, and real-world knowledge relevant to robotics and embodied AI scenarios.

Your final choice should be boxed in the following format:
$\\boxed{choice}$"""

def load_erqa_data(dataset: str, max_samples: int = None):
    """Load ERQA dataset from Hugging Face."""
    print(f"Loading ERQA dataset from Hugging Face: {dataset}")
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    
    dataset_obj = load_dataset(dataset, split="train", token=hf_token).shuffle(seed=42)
    examples = [dict(example) for example in dataset_obj] * 3
    
    # Limit samples if requested (before tripling)
    if max_samples:
        examples = examples[:max_samples]

    print(f"Loaded {len(dataset_obj)} unique examples (augmented to {len(examples)} total evaluations)")
    return examples

def format_message(example):
    """Format example into message format for the model."""
    question = example['question']
    base64_images = example['images_base64']
    
    # Build multimodal content
    content = [{"type": "text", "text": question}]
    
    for img_b64 in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f'data:image/png;base64,{img_b64}',
                "detail": "high"
            }
        })
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return messages


def parse_answer(response_text: str, ground_truth: str):
    """Parse and evaluate answer from response."""
    # Extract answer from boxed format: $\boxed{A}$ or \boxed{A}
    pattern = r'\\boxed\{([A-D])\}'
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    
    if not matches:
        return None  # No valid answer found, trigger retry
    
    extracted_answer = matches[-1].upper()
    is_correct = extracted_answer == ground_truth.strip().upper()
    
    return {
        'extracted_answer': extracted_answer,
        'is_correct': is_correct
    }

def compute_metrics(results):
    """Compute accuracy metrics from results."""
    total = len(results)
    correct = sum(r['is_correct'] for r in results)
    
    # Group by question type
    type_stats = {}
    for r in results:
        q_type = r['question_type']
        if q_type not in type_stats:
            type_stats[q_type] = {'correct': 0, 'total': 0}
        type_stats[q_type]['total'] += 1
        type_stats[q_type]['correct'] += r['is_correct']
    
    return {
        'accuracy': round(100 * correct / total, 2) if total > 0 else 0,
        'evaluated_questions': total,
        'type_accuracy': {k: round(100 * v['correct'] / v['total'], 2) 
                         for k, v in type_stats.items()},
    }

async def get_model_prediction(agent, example, example_idx, max_attempts: int = 3):
    """Get model prediction for a single example."""
    messages = format_message(example)
    
    # Try to get valid response with retries
    for attempt in range(max_attempts):
        try:
            response = await agent.async_completions(messages=messages)
            content = response.content
            assert content is not None, "Model returned None content"
            
            # Parse and evaluate the answer
            parse_result = parse_answer(content, example['answer'])
            if parse_result is None:
                raise ValueError("Response does not contain properly formatted answer")
            
            # Return result with all metadata
            return {
                **example,
                'example_idx': example_idx,
                'response': content,
                'extracted_answer': parse_result['extracted_answer'],
                'is_correct': parse_result['is_correct'],
                'num_images': len(example['images_base64'])
            }
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"{max_attempts} attempts failed for example {example_idx}: {e}")
                return None
    
    return None


async def generate_predictions(agent,
                                dataset: str,
                                max_concurrent: int = 10,
                                max_samples: int = None):
    """Generate model predictions for ERQA dataset."""
    
    # Load ERQA dataset
    examples = load_erqa_data(dataset, max_samples=max_samples)
    
    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def predict_with_semaphore(example, idx):
        async with semaphore:
            return await get_model_prediction(agent, example, idx)
    
    # Process all examples
    print(f"Generating predictions for {len(examples)} examples...")
    tasks = [predict_with_semaphore(example, idx) for idx, example in enumerate(examples)]
    
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating predictions"):
        result = await task
        if result:
            results.append(result)
    
    return results


def run_eval(model: str,
             output_file: str = None,
             dataset: str = "justinphan3110/erqa",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 10,
             max_samples: int = None):
    """
    Run ERQA evaluation.
    
    Args:
        model: Model name from models.yaml
        output_file: Path to output file (required)
        dataset: HuggingFace dataset identifier
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        max_samples: If set, limit evaluation to first N samples
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_agent = get_llm_agent_class(**get_agent_config(model, models_config))
    
    # Generate predictions
    results = asyncio.run(
        generate_predictions(
            agent=model_agent,
            dataset=dataset,
            max_concurrent=max_concurrent,
            max_samples=max_samples
        )
    )
    
    # Compute metrics
    metrics = compute_metrics(results)
    
    print("\n=== ERQA Results ===")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"Evaluated: {metrics['evaluated_questions']} questions")
    print("\nBy question type:")
    for q_type, acc in metrics['type_accuracy'].items():
        print(f"  {q_type}: {acc}%")
    
    print("\n===== Token Usage =====")
    print(f"Model: {model_agent.all_token_usage} | Max: {model_agent.max_token_usage}")
    
    # Save results
    print(f"\nSaving results to {output_path}")
    
    # Remove heavy/non-serializable fields before saving
    for result in results:
        result.pop("images_base64", None)
        result.pop("image_preview", None)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved successfully!")

if __name__ == '__main__':
    fire.Fire(run_eval)