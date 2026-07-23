#!/usr/bin/env python3
"""
Enigma evaluation script adapted for the leaderboard harness.
Integrates with shared LLM agents and models configuration.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
import re
import numpy as np
import yaml
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from dotenv import load_dotenv

from shared.llm_agents import get_llm_agent_class
from .enigmaeval_utils import (
    AnnotatedPuzzle,
    SPLITS,
    SOURCES,
    fetch_puzzles,
    PromptMaker
)

# Load HF_TOKEN (and any API keys) from .env so the gated cais/EnigmaEval
# dataset can be fetched. Users must accept the dataset conditions on the Hub.
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def standardize(input_string: str) -> str:
    """Standardize answer strings for comparison."""
    if input_string is np.nan or input_string is None:
        return ""
    
    input_string = str(input_string).lower()
    if "," in input_string:
        return set([standardize(s) for s in input_string.split(",")])
    else:
        return re.sub(r"[^a-zA-Z0-9]", "", input_string)

def load_models_config(config_path: str) -> dict:
    """Load models configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_config(model_name: str, models_config: dict) -> tuple:
    """Get model configuration and return model_name and generation_config."""
    if model_name not in models_config:
        raise ValueError(f"Model {model_name} not found in models config")
    
    config = models_config[model_name]
    full_model_name = config['model']
    generation_config = config.get('generation_config', {})
    
    return full_model_name, generation_config

def extract_answer_from_response(response: str) -> str:
    """Extract the answer from the model response using XML tags."""
    if not isinstance(response, str):
        return None
    
    # Extract from <answer></answer> XML tags
    xml_match = re.search(r'<answer>(.*?)</answer>', response, re.IGNORECASE | re.DOTALL)
    if xml_match:
        answer = xml_match.group(1).strip()
        return answer
    
    return None

def evaluate_answer(gt_answer: str, model_answer: str) -> bool:
    """Evaluate if the model answer matches the ground truth."""
    if model_answer is None:
        return False
    
    if "|" in gt_answer:
        # Multiple acceptable answers
        return standardize(model_answer) in [standardize(a) for a in gt_answer.split("|")]
    else:
        # Single answer
        return standardize(model_answer) == standardize(gt_answer)

def read_jsonl_records(path: str) -> list[dict]:
    """Read puzzle result records from a JSONL file (skips metrics summary lines)."""
    records = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get('_type') == 'metrics_summary':
                continue
            if 'puzzle_id' in entry:
                records.append(entry)
    return records


async def get_puzzle_prediction(llm_agent, record: dict):
    """Get and grade the model prediction for a single puzzle.

    Returns a result dict. On failure, model_response is None so the record is
    retryable on a later resume run.
    """
    messages = record["messages"]
    puzzle_id = record["puzzle_id"]
    gt_answer = record["gt_answer"]

    content = None
    for attempt in range(1):
        try:
            response = await llm_agent.async_completions(messages)
            content = response.content or response.reasoning_content or ""
            break
        except Exception as e:
            logger.error(f"Error processing puzzle '{puzzle_id}' (attempt {attempt + 1}/1): {e}")

    if content is None:
        # Prediction failed - record as retryable (model_response None)
        return {
            "puzzle_id": puzzle_id,
            "puzzle_source": record["puzzle_source"],
            "gt_answer": gt_answer,
            "model_response": None,
            "model_answer": None,
            "is_correct": False,
        }

    model_answer = extract_answer_from_response(content)
    is_correct = evaluate_answer(gt_answer, model_answer)
    return {
        "puzzle_id": puzzle_id,
        "puzzle_source": record["puzzle_source"],
        "gt_answer": gt_answer,
        "model_response": content,
        "model_answer": model_answer,
        "is_correct": is_correct,
    }


async def grade_puzzles(
    puzzles: list[AnnotatedPuzzle],
    llm_agent,
    output_file: str,
    max_concurrent: int = 32,
    text_only: bool = False,
    exclude_meta: bool = False,
    existing_results: list = None,
):
    """Grade puzzles using the LLM agent.

    Results are written incrementally to the JSONL output_file as they complete,
    so a killed job keeps everything done so far.
    """

    # Initialize prompt maker
    prompt_templates_dir = Path(__file__).parent / "prompt_templates"
    prompt_maker = PromptMaker(str(prompt_templates_dir))

    # Process each puzzle to create per-puzzle records (with pre-built messages)
    records = []
    for puzzle in puzzles:
        if exclude_meta and "meta" in puzzle.puzzle_id.lower():
            continue

        if text_only and puzzle.base64_images:
            continue

        # Generate prompts based on puzzle type
        if puzzle.puzzle_text is None:
            text_prompt, system_prompt = prompt_maker.get_prompt_pdf(
                puzzle.answer, prev_answers=puzzle.prev_answers
            )
        elif puzzle.puzzle_source == "MIT Mystery Hunt":
            text_prompt, system_prompt = prompt_maker.get_prompt_mit_transcribed(
                puzzle.puzzle_text, puzzle.answer
            )
        else:
            text_prompt, system_prompt = prompt_maker.get_prompt_standard_transcribed(
                puzzle.puzzle_text, puzzle.answer, prev_answers=puzzle.prev_answers
            )

        # Build messages directly
        messages = [{"role": "system", "content": system_prompt}]
        # Build user message content
        content = [{"type": "text", "text": text_prompt}]

        # Add images if present
        if puzzle.base64_images:
            content.extend([
                {"type": "image_url", "image_url": {"url": img_b64}}
                for img_b64 in puzzle.base64_images
            ])

        messages.append({"role": "user", "content": content})

        records.append({
            "puzzle_id": puzzle.puzzle_id,
            "puzzle_source": puzzle.puzzle_source,
            "gt_answer": puzzle.answer,
            "messages": messages,
        })

    total_puzzles = len(records)

    # If resuming, skip puzzles that already have a good (non-empty) response
    existing_by_id = {}
    if existing_results:
        for r in existing_results:
            existing_by_id[r["puzzle_id"]] = r
        good_ids = {r["puzzle_id"] for r in existing_results if r.get("model_response")}
        records_to_run = [rec for rec in records if rec["puzzle_id"] not in good_ids]
        empty = len(existing_by_id) - len(good_ids)
        print(
            f"Skipping {len(good_ids)} completed, rerunning {len(records_to_run)} "
            f"(empty/failed: {empty}, missing: {len(records_to_run) - empty})"
        )
        records = records_to_run

    if not records:
        logger.info("No puzzles to process")
    else:
        logger.info(f"Processing {len(records)} puzzles")

    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    async def predict_with_semaphore(record):
        async with semaphore:
            return await get_puzzle_prediction(llm_agent, record)

    tasks = [predict_with_semaphore(rec) for rec in records]

    # Open the JSONL file in append mode for incremental writes
    output_path = Path(output_file)
    jsonl_file = open(output_path, 'a')
    write_lock = asyncio.Lock()

    correct = 0
    processed = 0
    num_failed = 0
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
    for task in pbar:
        result = await task

        # Write result to JSONL immediately (retryable if model_response is None)
        async with write_lock:
            jsonl_file.write(json.dumps(result) + '\n')
            jsonl_file.flush()

        processed += 1
        correct += int(result["is_correct"])
        num_failed += int(result.get("model_response") is None)
        accuracy = 100 * correct / processed if processed > 0 else 0.0
        cost = llm_agent.all_token_usage.cost
        pbar.set_postfix({"acc": f"{accuracy:.1f}%", "cost": f"${cost:.3f}", "failed": num_failed})

    jsonl_file.close()

    # Read back the JSONL to build the authoritative merged result set
    # (dedupe by puzzle_id, last write wins so retried rows overwrite stale ones)
    merged_by_id = {}
    for r in read_jsonl_records(output_file):
        merged_by_id[r["puzzle_id"]] = r
    all_results = list(merged_by_id.values())

    return all_results, total_puzzles

def str2bool(v):
    """Parse a boolean from a CLI string (so --redo=False works like HLE)."""
    if isinstance(v, bool):
        return v
    return str(v).lower() not in ("false", "0", "no", "")


def get_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(description="Enigma evaluation with shared LLM agents")
    
    # Model configuration
    parser.add_argument("--model", type=str, required=True, help="Model name from models config")
    parser.add_argument("--models_config", type=str, required=True, help="Path to models config YAML")
    
    # Dataset configuration
    parser.add_argument("--split", "-s", type=str, default="all", help="Which puzzle sets to evaluate")
    parser.add_argument("--num_puzzles", "-n", type=int, default=None, help="Only run on first n puzzles")
    
    # Output configuration
    parser.add_argument("--output_dir", "-d", type=str, required=True, help="Output directory")
    
    # Processing options
    parser.add_argument("--raw", action="store_true", help="Use raw PDF")
    parser.add_argument("--exclude_meta", action="store_true", help="Exclude meta-puzzles")
    parser.add_argument("--text_only", action="store_true", help="Only run on text puzzles")
    parser.add_argument("--collated_pdf", action="store_true", help="Use collated PDF")
    
    # Execution options
    parser.add_argument("--redo", type=str2bool, default=True,
                       help="If True (default), rerun all puzzles. If False, resume: skip "
                            "puzzles with a good response and only rerun missing/empty/failed ones")
    parser.add_argument("--max_concurrent", type=int, default=32, help="Maximum concurrent requests")
    
    return parser

async def main():
    """Main evaluation function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Load models configuration
    models_config = load_models_config(args.models_config)
    full_model_name, generation_config = get_model_config(args.model, models_config)
    
    # Create LLM agent
    llm_agent = get_llm_agent_class(full_model_name, generation_config)
    logger.info(f"Using model: {full_model_name}")
    
    # Fetch puzzles
    puzzles = fetch_puzzles(args)
    logger.info(f"Loaded {len(puzzles)} puzzles")
    
    if args.num_puzzles is not None:
        puzzles = puzzles[:args.num_puzzles]
        logger.info(f"Limited to first {args.num_puzzles} puzzles")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create save path (JSONL for incremental writes)
    suffix = f"{'_raw' if args.raw else ''}{'_coll' if args.collated_pdf else ''}"
    save_path = os.path.join(args.output_dir, f"{args.model}{suffix}.jsonl")
    old_json_path = os.path.join(args.output_dir, f"{args.model}{suffix}.json")

    output_path = Path(save_path)

    # Load existing results for resume (redo=False)
    existing_results = None
    if args.redo:
        # Fresh run: start from an empty JSONL so we don't append to stale rows
        if output_path.exists():
            output_path.unlink()
    else:
        if output_path.exists():
            existing_results = read_jsonl_records(save_path)
            print(f"Loading existing results from {save_path}: {len(existing_results)} entries")
        elif os.path.exists(old_json_path):
            # Backward-compat: convert a legacy single-big-JSON result into the resume set
            try:
                with open(old_json_path, 'r') as f:
                    existing_results = json.load(f)
                print(f"Converting legacy JSON {old_json_path}: {len(existing_results)} entries")
            except (json.JSONDecodeError, ValueError):
                print(f"Could not parse legacy JSON {old_json_path}; starting fresh")
                existing_results = None

        # Rewrite the JSONL to contain only good results (drops stale empty/failed rows,
        # de-dupes by puzzle_id, and migrates legacy JSON into JSONL form)
        if existing_results:
            good_by_id = {}
            for r in existing_results:
                if r.get("model_response"):
                    good_by_id[r["puzzle_id"]] = r
            with open(output_path, 'w') as f:
                for r in good_by_id.values():
                    f.write(json.dumps(r) + '\n')
            print(f"Rewrote {save_path} with {len(good_by_id)} good results for resume")

    # Run evaluation (results are written incrementally to JSONL)
    all_results, total_puzzles = await grade_puzzles(
        puzzles=puzzles,
        llm_agent=llm_agent,
        output_file=save_path,
        max_concurrent=args.max_concurrent,
        text_only=args.text_only,
        exclude_meta=args.exclude_meta,
        existing_results=existing_results,
    )

    # Print evaluation results (denominator = total puzzles; failed/missing count as incorrect)
    total_correct = sum(int(result["is_correct"]) for result in all_results)
    denom = total_puzzles if total_puzzles > 0 else len(all_results)
    accuracy = total_correct / denom if denom > 0 else 0

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Overall accuracy: {accuracy:.3f} ({total_correct}/{denom})")

    # Print accuracy by source
    source_stats = {}
    for result in all_results:
        source = result["puzzle_source"]
        if source not in source_stats:
            source_stats[source] = {"correct": 0, "total": 0}
        source_stats[source]["total"] += 1
        if result["is_correct"]:
            source_stats[source]["correct"] += 1

    print("\nAccuracy by source:")
    for source, stats in sorted(source_stats.items()):
        source_accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        print(f"  {source:<25}: {source_accuracy:.3f} ({stats['correct']:>3}/{stats['total']:<3})")
    print("=" * 80)

    # Append a metrics summary line to the JSONL (authoritative dashboard number)
    metrics_line = {
        "_type": "metrics_summary",
        "model": args.model,
        "accuracy": round(accuracy, 4),
        "total_correct": total_correct,
        "total_puzzles": denom,
        "source_stats": source_stats,
    }
    with open(output_path, 'a') as f:
        f.write(json.dumps(metrics_line) + '\n')

    # Print token usage
    print("\n" + "=" * 80)
    print("TOKEN USAGE")
    print("=" * 80)
    print(f"  Total input tokens:  {llm_agent.all_token_usage.input_tokens:,}")
    print(f"  Total output tokens: {llm_agent.all_token_usage.output_tokens:,}")
    print(f"  Total tokens:        {llm_agent.all_token_usage.total_tokens:,}")
    print(f"  Cached tokens:       {llm_agent.all_token_usage.cached_tokens:,}")
    print(f"  Max input tokens:    {llm_agent.max_token_usage.input_tokens:,}")
    print(f"  Max output tokens:   {llm_agent.max_token_usage.output_tokens:,}")
    print(f"  Max total tokens:    {llm_agent.max_token_usage.total_tokens:,}")
    print("=" * 80)
    
    logger.info(f"Results saved to: {save_path}")
    
    print("\n" + "=" * 80)
    print("ENIGMAEVAL EVALUATION COMPLETE!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    asyncio.run(main())
