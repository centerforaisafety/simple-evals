#!/usr/bin/env python3
"""
MASK Benchmark Evaluation Script

This script runs the complete MASK (Model Alignment between Statements and Knowledge) benchmark,
which evaluates honesty in large language models by testing whether they contradict their own 
beliefs when pressured to lie.

The evaluation consists of four stages:
1. Download data from HuggingFace
2. Generate model responses (belief elicitation + lying prompts)
3. Evaluate responses using a judge model
4. Calculate metrics (honesty, accuracy)
"""

import argparse
import asyncio
import os
import sys
import glob
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import yaml
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio
from datasets import load_dataset

# Add parent directory to path to import shared modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.llm_agents import get_llm_agent_class
from mask_eval.evaluate_helpers import extract_boxed_evaluation

load_dotenv()


def load_mask_dataset(test_mode=False):
    """Load MASK dataset from HuggingFace and keep in memory."""
    print("=" * 80)
    print("STAGE 1: Loading MASK dataset from HuggingFace...")
    print("=" * 80)
    
    splits = ["continuations", "disinformation", "doubling_down_known_facts", 
              "known_facts", "provided_facts", "statistics"]
    
    all_data = {}
    
    for split in splits:
        ds = load_dataset("cais/mask", split)
        data = [dict(item) for item in ds['test']]
        
        if test_mode:
            # For test mode, only use 5 samples
            data = data[:min(5, len(data))]
        
        all_data[split] = data
        tqdm.write(f"  Loaded {len(data)} rows for {split}")
    
    return all_data


async def generate_responses(agent, lie_k, output_dir, dataset):
    """Generate model responses for MASK prompts."""
    print("=" * 80)
    print(f"STAGE 2: Generating responses with model: {agent.model}")
    print("=" * 80)
    
    # Create responses directory
    responses_dir = os.path.join(output_dir, "responses")
    os.makedirs(responses_dir, exist_ok=True)
    
    results = {}
    
    for split_name, data in tqdm(dataset.items(), desc="Processing splits"):
        # Determine belief_k based on archetype
        belief_k = {'1': 3} if 'statistics' in split_name else {'1': 3, '2': 1, '3': 1}
        
        # Create output filename
        modelname = agent.model.replace('/', '_')
        output_file = os.path.join(responses_dir, f'{split_name}_{modelname}.json')
        
        # Skip if already exists
        if os.path.exists(output_file):
            tqdm.write(f'Skipping {split_name} (already exists)')
            with open(output_file, 'r') as f:
                results[split_name] = json.load(f)
            continue
        
        tqdm.write(f'\nProcessing {split_name}')
        processed_data = await process_data(
            agent=agent,
            data=data,
            split_name=split_name,
            belief_k=belief_k,
            lie_k=lie_k
        )
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        results[split_name] = processed_data
    
    return results, agent


async def process_data(agent, data, split_name, belief_k, lie_k):
    """Process data and generate responses using the LLM agent."""
    # Make a copy to avoid modifying original
    data = [dict(item) for item in data]
    
    # Add belief_system_1 if not present
    for item in data:
        if 'belief_system_1' not in item and 'belief_elicit_1' in item:
            item['belief_system_1'] = 'You are a helpful assistant.'
    
    # Check if this is a doubling down scenario
    is_doubling_down = data and 'dd_turn_1_prompt' in data[0] and 'dd_turn_1_response' in data[0]
    
    # Prepare all prompts
    all_prompts = []
    indices = []
    prompt_types = []
    
    for index, item in enumerate(data):
        # Handle lying prompts
        if is_doubling_down:
            for run in range(lie_k):
                all_prompts.append([
                    {'role': 'system', 'content': item['system_prompt']},
                    {'role': 'user', 'content': item['dd_turn_1_prompt']},
                    {'role': 'assistant', 'content': item['dd_turn_1_response']},
                    {'role': 'user', 'content': item['user_prompt']},
                ])
                indices.append(index)
                prompt_types.append(f'lying_run{run+1}')
        else:
            for run in range(lie_k):
                all_prompts.append([
                    {'role': 'system', 'content': item['system_prompt']},
                    {'role': 'user', 'content': item['user_prompt']}
                ])
                indices.append(index)
                prompt_types.append(f'lying_run{run+1}')
        
        # Handle belief elicitation prompts
        if 'belief_elicit_1' in item:
            for be_num, k_value in belief_k.items():
                belief_key = f'belief_elicit_{be_num}'
                if item.get(belief_key):
                    for run in range(k_value):
                        all_prompts.append([
                            {'role': 'system', 'content': item['belief_system_1']},
                            {'role': 'user', 'content': item[belief_key]}
                        ])
                        indices.append(index)
                        prompt_types.append(f'be{be_num}_run{run+1}')
    
    # Generate all responses with rate limiting
    print(f"Generating {len(all_prompts)} responses...")
    
    # Create semaphore for rate limiting (max 50 concurrent requests)
    semaphore = asyncio.Semaphore(50)
    
    async def generate_with_semaphore(prompt):
        async with semaphore:
            return await generate_single_response(agent, prompt)
    
    responses = await tqdm_asyncio.gather(*[
        generate_with_semaphore(prompt)
        for prompt in all_prompts
    ], desc=f"Generating responses for {split_name}")
    
    # Create column mapping
    column_mapping = {
        f'lying_run{run+1}': f'generation(System Prompt + User Prompt)_run{run+1}'
        for run in range(lie_k)
    }
    
    if data and 'belief_elicit_1' in data[0]:
        for be_num, k_value in belief_k.items():
            for run in range(k_value):
                column_mapping[f'be{be_num}_run{run+1}'] = f'generation(belief_system_1 + belief_elicit_{be_num})_run{run+1}'
    
    # Store responses in data
    for response, index, prompt_type in zip(responses, indices, prompt_types):
        field_name = column_mapping[prompt_type]
        data[index][field_name] = response
    
    return data


async def generate_single_response(agent, messages):
    """Generate a single response with retry logic."""
    max_retries = 7
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            response = await agent.async_completions(messages)
            return response.content
        except Exception as e:
            error_msg = str(e)
            if 'invalid_prompt' in error_msg or 'content_policy' in error_msg.lower():
                print(f"Content policy violation: {error_msg}")
                return "[Error: Prompt violated content policies]"
            elif attempt == max_retries - 1:
                print(f"Max retries exceeded. Final error: {error_msg}")
                return f"[ERROR: {error_msg}]"
            else:
                print(f"Error on attempt {attempt + 1}/{max_retries}: {error_msg}. Retrying in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay *= 2


async def evaluate_responses(judge_agent, max_concurrent, output_dir, responses_data):
    """Evaluate generated responses using a judge model."""
    print("=" * 80)
    print(f"STAGE 3: Evaluating responses with judge model: {judge_agent.model}")
    print("=" * 80)
    
    # Create evaluated directory
    evaluated_dir = os.path.join(output_dir, "evaluated")
    os.makedirs(evaluated_dir, exist_ok=True)
    
    results = {}
    
    for split_name, data in tqdm(responses_data.items(), desc="Evaluating splits"):
        output_file = os.path.join(evaluated_dir, f'{split_name}.json')
        
        if os.path.exists(output_file):
            tqdm.write(f"Skipping {split_name} (already evaluated)")
            with open(output_file, 'r') as f:
                results[split_name] = json.load(f)
            continue
        
        tqdm.write(f"\nEvaluating {split_name}...")
        evaluated_data = await evaluate_data(judge_agent, data, split_name, max_concurrent)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(evaluated_data, f, indent=2)
        
        results[split_name] = evaluated_data

    return results, judge_agent


async def evaluate_data(judge_agent, data, split_name, max_concurrent):
    """Evaluate data with responses."""
    from mask_eval.evaluate_helpers import (
        parse_proposition, 
        extract_boxed_evaluation,
        get_binary_judge_prompt,
        get_numerical_judge_prompt,
        get_doubling_down_prompt
    )
    
    # Make a copy to avoid modifying original
    data = [dict(item) for item in data]
    
    # Initialize evaluation fields
    for item in data:
        item['ground_truth_extracted'] = item.get('formatted_ground_truth')
    
    # Identify all generation fields
    if data:
        sample_keys = data[0].keys()
        system_user_run_cols = [k for k in sample_keys if k.startswith('generation(System Prompt + User Prompt)_run')]
        
        # Initialize eval fields
        for item in data:
            for run_col in system_user_run_cols:
                eval_col = run_col.replace('generation', 'eval')
                item[eval_col] = None
                item[f'{eval_col}_extracted'] = None
        
        # Handle belief columns
        belief_columns = []
        for i in range(1, 5):
            belief_col = f'belief_elicit_{i}'
            if belief_col in sample_keys:
                run_cols = [k for k in sample_keys if k.startswith(f'generation(belief_system_1 + belief_elicit_{i})_run')]
                for run_col in run_cols:
                    eval_col = run_col.replace('generation', 'eval')
                    for item in data:
                        item[eval_col] = None
                        item[f'{eval_col}_extracted'] = None
                    belief_columns.append((belief_col, run_col))
    
    # Process each row
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_row_with_semaphore(idx, item):
        async with semaphore:
            return await evaluate_row(judge_agent, idx, item, data, split_name, belief_columns, system_user_run_cols)
    
    tasks = [process_row_with_semaphore(idx, item) for idx, item in enumerate(data)]
    row_results = await tqdm_asyncio.gather(*tasks, desc=f"Evaluating {split_name}")
    
    # Apply results to data
    for idx, results in enumerate(row_results):
        for field_name, value in results.items():
            data[idx][field_name] = value
    
    return data


async def evaluate_row(judge_agent, idx, item, data, split_name, belief_columns, system_user_run_cols):
    """Evaluate a single row with all its responses."""
    from mask_eval.evaluate_helpers import (
        parse_proposition, 
        extract_boxed_evaluation,
        get_binary_judge_prompt,
        get_numerical_judge_prompt,
        get_doubling_down_prompt
    )
    
    results = {}
    
    # Gather all evaluation tasks for this row
    eval_tasks = []
    task_keys = []
    
    # Check if this is doubling down
    is_doubling_down = 'dd_turn_1_prompt' in item and 'dd_turn_1_response' in item
    
    # Evaluate system+user responses
    if system_user_run_cols:
        for run_col in system_user_run_cols:
            if not item.get(run_col):
                continue
                
            prop_type, prop_data = parse_proposition(item['proposition'])
            
            if prop_type == 'binary':
                prop1, prop2 = prop_data
                if is_doubling_down:
                    prompt = get_doubling_down_prompt(
                        item['system_prompt'],
                        item['dd_turn_1_prompt'],
                        item['dd_turn_1_response'],
                        item['user_prompt'],
                        item[run_col],
                        prop1, prop2
                    )
                else:
                    prompt = get_binary_judge_prompt(
                        item['system_prompt'],
                        item['user_prompt'],
                        item[run_col],
                        prop1, prop2,
                        filepath=split_name
                    )
                
                eval_tasks.append(evaluate_with_judge(judge_agent, prompt, is_numerical=False))
                task_keys.append(run_col)
            elif prop_type == 'numerical':
                # For numerical propositions, evaluate the system+user response
                prompt = get_numerical_judge_prompt(
                    item['system_prompt'],
                    item['user_prompt'],
                    item[run_col],
                    item['proposition']
                )
                eval_tasks.append(evaluate_with_judge(judge_agent, prompt, is_numerical=True))
                task_keys.append(run_col)
    
    # Evaluate belief responses
    prop_type, prop_data = parse_proposition(item['proposition'])
    
    if prop_type == 'numerical':
        # For numerical, combine all belief responses
        belief_responses = []
        for belief_col, gen_col in belief_columns:
            if item.get(belief_col) and item.get(gen_col):
                belief_responses.append((item[belief_col], item[gen_col]))
        
        if belief_responses:
            # Create combined prompt
            combined_text = "\n".join([
                f"Belief Prompt {i}:\n{prompt}\nResponse {i}:\n{response}\n---"
                for i, (prompt, response) in enumerate(belief_responses, 1)
            ])
            combined_text = "[Note: Slight change in plans, evaluate this collection of responses as a whole.]\n" + combined_text
            
            belief_system = item.get('belief_system_1', 'You are a helpful assistant.')
            prompt = get_numerical_judge_prompt(
                belief_system,
                item['belief_elicit_1'],
                combined_text,
                item['proposition']
            )
            
            eval_tasks.append(evaluate_with_judge(judge_agent, prompt, is_numerical=True))
            # Store result for all belief columns
            for _, gen_col in belief_columns:
                task_keys.append(gen_col)
    
    elif prop_type == 'binary':
        # For binary, evaluate each belief response individually
        prop1, prop2 = prop_data
        for belief_col, gen_col in belief_columns:
            if item.get(belief_col) and item.get(gen_col):
                belief_system = item.get('belief_system_1', 'You are a helpful assistant.')
                use_belief_prompt = 'direct' if 'elicit_1' in belief_col else 'consistency'
                
                prompt = get_binary_judge_prompt(
                    belief_system,
                    item[belief_col],
                    item[gen_col],
                    prop1, prop2,
                    use_belief_prompt=use_belief_prompt
                )
                
                eval_tasks.append(evaluate_with_judge(judge_agent, prompt, is_numerical=False))
                task_keys.append(gen_col)
    
    # Execute all tasks
    if eval_tasks:
        try:
            eval_results = await asyncio.wait_for(
                asyncio.gather(*eval_tasks, return_exceptions=True),
                timeout=120
            )
        except asyncio.TimeoutError:
            print(f"Timeout processing row {idx}")
            eval_results = ["TIMEOUT_ERROR"] * len(eval_tasks)
        
        # Store results
        if prop_type == 'numerical' and belief_columns and eval_results:
            # For numerical belief, store the same result for all belief columns
            belief_result = eval_results[len(system_user_run_cols):len(system_user_run_cols)+1][0] if len(eval_results) > len(system_user_run_cols) else None
            
            # Store system+user results
            for i, run_col in enumerate(system_user_run_cols):
                if i < len(eval_results):
                    eval_col = run_col.replace('generation', 'eval')
                    results[eval_col] = eval_results[i]
                    extracted = extract_boxed_evaluation(eval_results[i])
                    results[f'{eval_col}_extracted'] = str(extracted) if extracted else None
            
            # Store belief results
            if belief_result:
                extracted_belief = extract_boxed_evaluation(belief_result)
                for _, gen_col in belief_columns:
                    eval_col = gen_col.replace('generation', 'eval')
                    results[eval_col] = belief_result
                    results[f'{eval_col}_extracted'] = str(extracted_belief) if extracted_belief else None
        else:
            # Store results normally
            for task_key, result in zip(task_keys, eval_results):
                eval_col = task_key.replace('generation', 'eval')
                results[eval_col] = result
                extracted = extract_boxed_evaluation(result)
                results[f'{eval_col}_extracted'] = str(extracted) if extracted else None
    
    return results


async def evaluate_with_judge(judge_agent, prompt, is_numerical=False):
    """Evaluate a single prompt with the judge model."""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': prompt}
            ]
            
            response = await judge_agent.async_completions(messages)
            content = response.content
            
            # Check if we can extract the answer
            if extract_boxed_evaluation(content) is not None:
                return content
            
            return content
        
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2
            else:
                print(f"Error after {max_retries} attempts: {str(e)}", flush=True)
                return f"ERROR: {str(e)}"
    
    return "ERROR: Max retries exceeded"


def calculate_metrics(output_dir, evaluated_data, model_name):
    """Calculate honesty and accuracy metrics."""
    print("=" * 80)
    print("STAGE 4: Calculating metrics...")
    print("=" * 80)
    
    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Import metric calculation functions
    from mask_eval.metric import process_evaluated_data
    from mask_eval.process_metrics import process_json_files
    
    # Calculate metrics from evaluated data
    metrics_data = process_evaluated_data(evaluated_data, metrics_dir, model_name)
    
    # Process final metrics
    results = process_json_files(metrics_dir, test_mode=False)
    
    output_json = os.path.join(metrics_dir, "all_results.json")
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMetrics saved to {output_json}")
    print("\n" + "=" * 80)
    print("FINAL RESULTS:")
    print("=" * 80)
    print(json.dumps(results, indent=2))
    print("=" * 80)
    
    # Print aggregate summary
    print("\n" + "=" * 80)
    print("AGGREGATE HONESTY SUMMARY")
    print("=" * 80)
    
    total_responses = 0
    weighted_honesty = 0
    weighted_accuracy = 0
    accuracy_responses = 0
    
    for archetype, archetypes_data in results.items():
        for subtype, metrics in archetypes_data.items():
            n = metrics.get('total_responses', 0)
            if n > 0:
                total_responses += n
                if 'honesty_score_1' in metrics:
                    weighted_honesty += metrics['honesty_score_1'] * n
                if 'accurate' in metrics:
                    weighted_accuracy += metrics['accurate'] * n
                    accuracy_responses += n
    
    if total_responses > 0:
        avg_honesty = weighted_honesty / total_responses
        avg_deception = 100 - avg_honesty
        print(f"\n📊 Overall Honesty Score: {avg_honesty:.2f}%")
        print(f"   CAIS Safety Index Deception (100-honest) = {avg_deception:.2f}%")
        print(f"   (Based on {total_responses} total responses)")
        
        if accuracy_responses > 0:
            avg_accuracy = weighted_accuracy / accuracy_responses
            print(f"\n📊 Overall Accuracy: {avg_accuracy:.2f}%")
            print(f"   (Based on {accuracy_responses} responses with ground truth)")
        
        print("\n📋 Breakdown by Archetype:")
        for archetype, archetypes_data in results.items():
            for subtype, metrics in archetypes_data.items():
                archetype_name = f"{archetype}_{subtype}" if subtype != "facts" else archetype
                n = metrics.get('total_responses', 0)
                h = metrics.get('honesty_score_1', 0)
                a = metrics.get('accurate', 'N/A')
                if a != 'N/A':
                    print(f"  • {archetype_name:25s}: Honesty={h:5.1f}%  Accuracy={a:5.1f}%  (n={n})")
                else:
                    print(f"  • {archetype_name:25s}: Honesty={h:5.1f}%  (n={n})")
    
    print("=" * 80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="MASK Benchmark Evaluation")
    parser.add_argument("--model", default="gpt-4o", help="Model to evaluate")
    parser.add_argument("--output_dir", required=True, help="Output directory for results")
    parser.add_argument("--models_config", default="configs/models.yaml", 
                       help="Path to models config file")
    parser.add_argument("--judge_model", default="openai/gpt-5-mini",
                       help="Judge model for evaluation (default: openai/gpt-5-mini)")
    parser.add_argument("--lie_k", type=int, default=1, 
                       help="Number of lying responses per prompt")
    parser.add_argument("--max_concurrent", type=int, default=50, 
                       help="Concurrent API calls for evaluation")
    parser.add_argument("--test", action="store_true", 
                       help="Run in test mode with small sample")
    parser.add_argument("--skip_generate", action="store_true", 
                       help="Skip response generation")
    parser.add_argument("--skip_evaluate", action="store_true", 
                       help="Skip response evaluation")
    parser.add_argument("--skip_metrics", action="store_true", 
                       help="Skip metrics calculation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model configuration and initialize generation agent
    with open(args.models_config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.model not in config:
        raise ValueError(f"Model {args.model} not found in {args.models_config}")
    
    model_config = config[args.model]
    judge_model_config = config[args.judge_model]
    model_name = model_config.get('model')
    generation_config = model_config.get('generation_config', {})


    judge_model_name = judge_model_config.get('model')
    judge_generation_config = judge_model_config.get('generation_config', {})

    print(f"Initializing generation agent: {model_name}")
    generation_agent = get_llm_agent_class(model_name, generation_config=generation_config)
    
    print(f"Initializing judge agent: {args.judge_model}")
    judge_agent = get_llm_agent_class(judge_model_name, generation_config=judge_generation_config)
    
    print("\n" + "=" * 80)
    print("MASK BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Generation model: {generation_agent.model}")
    print(f"Judge model: {judge_agent.model}")
    print(f"Output directory: {args.output_dir}")
    print(f"Test mode: {args.test}")
    print(f"Lie responses per prompt: {args.lie_k}")
    print(f"Max concurrent: {args.max_concurrent}")
    print("=" * 80 + "\n")
    
    # Stage 1: Load dataset
    dataset = load_mask_dataset(args.test)
    
    # Stage 2: Generate responses
    responses_data = None
    if not args.skip_generate:
        responses_data, generation_agent = asyncio.run(generate_responses(
            agent=generation_agent,
            lie_k=args.lie_k,
            output_dir=args.output_dir,
            dataset=dataset
        ))
    else:
        # Load existing responses
        responses_dir = os.path.join(args.output_dir, "responses")
        if os.path.exists(responses_dir):
            responses_data = {}
            for file in glob.glob(os.path.join(responses_dir, "*.json")):
                split_name = os.path.basename(file).replace(f"_{generation_agent.model.replace('/', '_')}.json", "")
                with open(file, 'r') as f:
                    responses_data[split_name] = json.load(f)
            print(f"Loaded {len(responses_data)} response files from {responses_dir}")
    
    # Stage 3: Evaluate responses
    evaluated_data = None
    if not args.skip_evaluate:
        if responses_data is None:
            raise ValueError("Cannot evaluate without responses. Don't use --skip_generate or provide existing responses.")
        evaluated_data, judge_agent = asyncio.run(evaluate_responses(
            judge_agent=judge_agent,
            max_concurrent=args.max_concurrent,
            output_dir=args.output_dir,
            responses_data=responses_data
        ))
    else:
        # Load existing evaluated data
        evaluated_dir = os.path.join(args.output_dir, "evaluated")
        if os.path.exists(evaluated_dir):
            evaluated_data = {}
            for file in glob.glob(os.path.join(evaluated_dir, "*.json")):
                split_name = os.path.basename(file).replace(".json", "")
                with open(file, 'r') as f:
                    evaluated_data[split_name] = json.load(f)
            print(f"Loaded {len(evaluated_data)} evaluated files from {evaluated_dir}")
    
    # Stage 4: Calculate metrics
    if not args.skip_metrics:
        if evaluated_data is None:
            raise ValueError("Cannot calculate metrics without evaluated data. Don't use --skip_evaluate or provide existing evaluated data.")
        calculate_metrics(args.output_dir, evaluated_data, args.model)
    
    # Print total token usage
    print("\n" + "=" * 80)
    print("TOTAL TOKEN USAGE SUMMARY")
    print("=" * 80)
    
    if generation_agent:
        print(f"\nGeneration Model ({generation_agent.model}):")
        print(f"  Total input tokens:  {generation_agent.all_token_usage.input_tokens:,}")
        print(f"  Total output tokens: {generation_agent.all_token_usage.output_tokens:,}")
        print(f"  Total tokens:        {generation_agent.all_token_usage.total_tokens:,}")
        print(f"  Cached tokens:       {generation_agent.all_token_usage.cached_tokens:,}")
    
    if judge_agent:
        print(f"\nJudge Model ({judge_agent.model}):")
        print(f"  Total input tokens:  {judge_agent.all_token_usage.input_tokens:,}")
        print(f"  Total output tokens: {judge_agent.all_token_usage.output_tokens:,}")
        print(f"  Total tokens:        {judge_agent.all_token_usage.total_tokens:,}")
        print(f"  Cached tokens:       {judge_agent.all_token_usage.cached_tokens:,}")
    
    
    print("\n" + "=" * 80)
    print("MASK EVALUATION COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

