#!/usr/bin/env python3
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
import base64

load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared import get_llm_agent_class, get_agent_config

# =============== PROMPTS ===============
SYSTEM_PROMPT = """Please provide your answer in the following format:
$\\boxed{choice}$

For example: $\\boxed{A}$"""

# Path to images directory
IMAGES_DIR = Path(__file__).parent / "images"


def download_images_if_needed():
    """Download and extract images if not already present."""
    # Check if images directory exists and has content
    if IMAGES_DIR.exists() and any(IMAGES_DIR.iterdir()):
        return
    
    print("Downloading SpatialViz images (~150MB)...")
    import subprocess
    import zipfile
    
    script_dir = Path(__file__).parent
    zip_path = script_dir / "images.zip"
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download images
    url = "https://huggingface.co/datasets/PLM-Team/Spatial-Visualization-Benchmark/resolve/main/images.zip"
    subprocess.run(["wget", "-O", str(zip_path), url], check=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(IMAGES_DIR)
    zip_path.unlink()


def load_spatialviz_data(dataset: str, max_samples: int = None):
    """Load SpatialViz dataset from Hugging Face."""
    print(f"Loading SpatialViz dataset from Hugging Face: {dataset}")
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    
    dataset_obj = load_dataset(dataset, split="test", token=hf_token).shuffle(seed=42)
    examples = [dict(example) for example in dataset_obj]
    
    # Limit samples if requested
    if max_samples:
        examples = examples[:max_samples]
    
    print(f"Loaded {len(examples)} examples")
    return examples


def load_and_encode_image(image_path: Path) -> str:
    """Load image and encode as base64."""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def format_message(example):
    """Format example into message format for the model."""
    # Build the question with choices
    question_text = example['Question']
    choices = example['Choices']
    
    # Format choices as A, B, C, D
    formatted_choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    full_question = f"{question_text}\n\n{formatted_choices}"
    
    # Load image
    image_id = example['Image_id']
    category = example['Category']
    task = example['Task']
    level = example['Level']
    
    # Image path format: {Category}/{Task}/{Level}/{Image_id}.png
    image_path = IMAGES_DIR / category / task / level / f"{image_id}.png"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Encode image
    image_b64 = load_and_encode_image(image_path)
    
    # Build multimodal content (image first, then question)
    content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f'data:image/png;base64,{image_b64}',
                "detail": "high"
            }
        },
        {
            "type": "text",
            "text": full_question
        }
    ]
    
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
    
    # Group by category
    category_stats = {}
    for r in results:
        category = r['Category']
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        category_stats[category]['correct'] += r['is_correct']
    
    # Group by task
    task_stats = {}
    for r in results:
        task = r['Task']
        if task not in task_stats:
            task_stats[task] = {'correct': 0, 'total': 0}
        task_stats[task]['total'] += 1
        task_stats[task]['correct'] += r['is_correct']
    
    return {
        'accuracy': round(100 * correct / total, 2) if total > 0 else 0,
        'correct': correct,
        'total': total,
        'category_accuracy': {k: round(100 * v['correct'] / v['total'], 2) 
                             for k, v in category_stats.items() if v['total'] > 0},
        'category_counts': {k: f"{v['correct']}/{v['total']}" 
                           for k, v in category_stats.items() if v['total'] > 0},
        'task_accuracy': {k: round(100 * v['correct'] / v['total'], 2) 
                         for k, v in task_stats.items() if v['total'] > 0},
        'task_counts': {k: f"{v['correct']}/{v['total']}" 
                       for k, v in task_stats.items() if v['total'] > 0}
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
            parse_result = parse_answer(content, example['Answer'])
            if parse_result is None:
                raise ValueError("Response does not contain properly formatted answer")
            
            # Return result with all metadata
            return {
                **example,
                'example_idx': example_idx,
                'response': content,
                'extracted_answer': parse_result['extracted_answer'],
                'is_correct': parse_result['is_correct'],
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
    """Generate model predictions for SpatialViz dataset."""
    
    # Load SpatialViz dataset
    examples = load_spatialviz_data(dataset, max_samples=max_samples)
    
    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def predict_with_semaphore(example, idx):
        async with semaphore:
            return await get_model_prediction(agent, example, idx)
    
    # Process all examples
    print(f"Generating predictions for {len(examples)} examples...")
    tasks = [predict_with_semaphore(example, idx) for idx, example in enumerate(examples)]
    
    results = []
    correct = 0
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
    for task in pbar:
        result = await task
        if result:
            results.append(result)
            correct += int(result['is_correct'])
            
            # Update progress bar with accuracy and cost
            accuracy = 100 * correct / len(results)
            cost = agent.all_token_usage.cost
            pbar.set_postfix({
                "acc": f"{accuracy:.1f}%",
                "cost": f"${cost:.3f}"
            })
    
    return results


def run_eval(model: str,
             output_file: str = None,
             dataset: str = "PLM-Team/Spatial-Visualization-Benchmark",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 32,
             max_samples: int = None):
    """
    Run SpatialViz evaluation.
    
    Args:
        model: Model name from models.yaml
        output_file: Path to output file (required)
        dataset: HuggingFace dataset identifier
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        max_samples: If set, limit evaluation to first N samples
    """
    # Download images if needed
    download_images_if_needed()
    
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
    
    print("\n=== SpatialViz Results ===")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {metrics['accuracy']}% ({metrics['correct']}/{metrics['total']})")
    print(f"Evaluated: {metrics['total']} examples")
    
    print("\nBy category:")
    for category, acc in sorted(metrics['category_accuracy'].items()):
        counts = metrics['category_counts'][category]
        print(f"  {category}: {acc}% ({counts})")
    
    print("\nBy task:")
    for task, acc in sorted(metrics['task_accuracy'].items()):
        counts = metrics['task_counts'][task]
        print(f"  {task}: {acc}% ({counts})")
    
    print("\n===== Token Usage =====")
    print(f"Total Cost: ${model_agent.all_token_usage.cost:.4f}")
    print(f"Tokens: {model_agent.all_token_usage}")
    print(f"Max Single Request: {model_agent.max_token_usage}")
    
    # Save results
    print(f"\nSaving results to {output_path}")
    
    # Save full results with metadata
    output_data = {
        'model': model,
        'dataset': dataset,
        'metrics': metrics,
        'results': results
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    print(f"Results saved successfully!")


if __name__ == '__main__':
    fire.Fire(run_eval)

