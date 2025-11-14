#!/usr/bin/env python3
"""
ARC-AGI-2 Evaluation Script
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
SYSTEM_PROMPT = """You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.

Respond in the format of the training output examples

--Training Examples--
{training_examples}
--End of Training Examples--

--Test Input--
{test_input}
--End of Test Input--

Your final answer output should be in the following format:
<answer>
...
</answer>
"""


def load_arc_agi_2_data(dataset: str, max_samples: int = None):
    """Load ARC-AGI-2 dataset from Hugging Face."""
    print(f"Loading ARC-AGI-2 dataset from Hugging Face: {dataset}")
    
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")
    
    dataset_obj = load_dataset(dataset, split="test", token=hf_token)
    examples = [dict(example) for example in dataset_obj]
    
    # Limit samples if requested
    if max_samples:
        examples = examples[:max_samples]
    
    print(f"Loaded {len(examples)} tasks")
    return examples


def grid_to_string(grid: list) -> str:
    """Convert a 2D grid to string format."""
    return "\n".join(json.dumps(row) for row in grid)


def format_task(example):
    """Format a task with training examples and test inputs."""
    fewshots = example["fewshots"]
    test_cases = example["question"]
    
    # Build training examples string
    training_examples = ""
    for i, pair in enumerate(fewshots):
        training_examples += f"--Example {i}--\n\nINPUT:\n\n"
        training_examples += grid_to_string(pair["input"]) + "\n\n"
        training_examples += "OUTPUT:\n\n<answer>\n"
        training_examples += grid_to_string(pair["output"])
        training_examples += "\n</answer>\n\n"
    
    # Process each test case
    test_pairs = []
    for pair_idx, test_case in enumerate(test_cases):
        test_input = grid_to_string(test_case["input"])
        
        prompt = SYSTEM_PROMPT.format(
            training_examples=training_examples,
            test_input=test_input
        )
        
        test_pairs.append({
            'pair_idx': pair_idx,
            'prompt': prompt,
            'ground_truth': test_case["output"],
            'test_input': test_case["input"]
        })
    
    return {
        'test_pairs': test_pairs,
        'num_fewshots': len(fewshots),
        'num_test_pairs': len(test_cases)
    }


def parse_answer(response_text: str):
    """Parse response to extract output grid from <answer></answer> tags."""
    # Extract content between <answer></answer> XML tags
    answer_pattern = r'<answer>\s*(.*?)\s*</answer>'
    answer_matches = re.findall(answer_pattern, response_text, re.DOTALL | re.IGNORECASE)
    
    if not answer_matches:
        return None
    
    match = answer_matches[-1].strip()
    if not match:
        return None
    
    # Try to extract JSON array
    array_pattern = r'\[.*\]'
    array_match = re.search(array_pattern, match, re.DOTALL)
    
    if array_match:
        try:
            parsed_grid = json.loads(array_match.group(0))
            if isinstance(parsed_grid, list):
                return parsed_grid
        except json.JSONDecodeError:
            pass
    
    # Fallback: try line-by-line parsing
    grid = []
    for line in match.split('\n'):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            try:
                row = json.loads(line)
                if isinstance(row, list):
                    grid.append(row)
            except json.JSONDecodeError:
                continue
    
    return grid if grid else None


def compute_metrics(task_results):
    """Compute task-level accuracy (official ARC-AGI metric)."""
    total_tasks = len(task_results)
    tasks_solved = 0
    total_pairs = 0
    valid_pairs = 0
    
    for task_result in task_results:
        any_pair_correct = False
        
        for pair_result in task_result['pair_results']:
            total_pairs += 1
            
            if pair_result.get('predicted_output') is not None:
                valid_pairs += 1
                if pair_result.get('is_correct', False):
                    any_pair_correct = True
        
        if any_pair_correct:
            tasks_solved += 1
    
    return {
        'task_accuracy': round(100 * tasks_solved / total_tasks, 2) if total_tasks > 0 else 0,
        'tasks_solved': tasks_solved,
        'total_tasks': total_tasks,
        'total_pairs': total_pairs,
        'valid_pairs': valid_pairs,
    }


async def evaluate_single_pair(task_idx, test_pair, agent, max_attempts: int = 3):
    """Evaluate a single test pair with retry logic."""
    pair_idx = test_pair['pair_idx']
    
    for attempt in range(max_attempts):
        try:
            messages = [{"role": "user", "content": test_pair['prompt']}]
            response = await agent.async_completions(messages=messages)
            
            content = response.content.strip() if response.content else ""
            predicted_output = parse_answer(content)
            
            if predicted_output is None:
                raise ValueError("Failed to parse output")
            
            return {
                "pair_idx": pair_idx,
                "response": content,
                "ground_truth": test_pair['ground_truth'],
                "predicted_output": predicted_output,
                "is_correct": predicted_output == test_pair['ground_truth'],
                "test_input": test_pair['test_input']
            }
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"Task {task_idx}, pair {pair_idx} failed after {max_attempts} attempts: {e}")
    
    # All attempts failed
    return {
        "pair_idx": pair_idx,
        "response": None,
        "ground_truth": test_pair['ground_truth'],
        "predicted_output": None,
        "is_correct": False,
        "test_input": test_pair['test_input']
    }


async def evaluate_single_task(task_idx, task_example, agent):
    """Evaluate a single task with all its test pairs."""
    task_data = format_task(task_example)
    
    pair_results = []
    for test_pair in task_data['test_pairs']:
        pair_result = await evaluate_single_pair(task_idx, test_pair, agent)
        pair_results.append(pair_result)
    
    return {
        "task_idx": task_idx,
        "num_fewshots": task_data['num_fewshots'],
        "num_test_pairs": task_data['num_test_pairs'],
        "pair_results": pair_results
    }


async def generate_predictions(agent,
                                dataset: str,
                                max_concurrent: int = 10,
                                max_samples: int = None):
    """Generate predictions for ARC-AGI-2 dataset."""
    
    # Load dataset
    examples = load_arc_agi_2_data(dataset, max_samples=max_samples)
    
    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def evaluate_with_semaphore(task_idx, task_example):
        async with semaphore:
            return await evaluate_single_task(task_idx, task_example, agent)
    
    # Process all tasks
    print(f"Evaluating {len(examples)} tasks...")
    tasks = [evaluate_with_semaphore(idx, example) for idx, example in enumerate(examples)]
    
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating tasks"):
        result = await task
        if result:
            results.append(result)
    
    return results


def run_eval(model: str,
             output_file: str = None,
             dataset: str = "arc-agi-community/arc-agi-2",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 10,
             max_samples: int = None):
    """
    Run ARC-AGI-2 evaluation.
    
    Args:
        model: Model name from models.yaml
        output_file: Path to output file (required)
        dataset: HuggingFace dataset identifier
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        max_samples: If set, limit evaluation to first N tasks
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
    
    print("\n=== ARC-AGI-2 Results ===")
    print(f"Dataset: {dataset}")
    print(f"Task Accuracy: {metrics['task_accuracy']}% ({metrics['tasks_solved']}/{metrics['total_tasks']})")
    print(f"Total pairs evaluated: {metrics['total_pairs']}")
    print(f"Valid pairs: {metrics['valid_pairs']}")
    
    print("\n===== Token Usage =====")
    print(f"Model: {model_agent.all_token_usage} | Max: {model_agent.max_token_usage}")
    
    # Save results
    print(f"\nSaving results to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved successfully!")


if __name__ == '__main__':
    fire.Fire(run_eval)
