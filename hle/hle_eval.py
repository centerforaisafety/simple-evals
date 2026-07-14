#!/usr/bin/env python3
"""
HLE (Human-Level Expertise) Evaluation Script
Evaluates models on their ability to answer expert-level questions with confidence calibration.
"""

import asyncio
import os
import json
import math
import argparse
import fire
import sys
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from shared import get_llm_agent_class, get_agent_config

# =============== PROMPTS ===============
SYSTEM_PROMPT = """Your response should be in the following format:
Explanation: {your explanation for your answer choice}
Answer: {your chosen answer}
Confidence: {your confidence score between 0% and 100% for your answer}"""

# Judge prompt for evaluating answers
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in XML format with the following tags:

<extracted_final_answer>The final exact answer extracted from the [response]. Put 'None' if there is no exact, final answer to extract from the response.</extracted_final_answer>

<reasoning>Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.</reasoning>

<correct>Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.</correct>

<confidence>The extracted confidence score between 0 and 100 from [response]. Put 100 if there is no confidence score available.</confidence>"""


def load_hle_data(dataset: str, max_samples: int = None, text_only: bool = False):
    """Load HLE dataset from Hugging Face."""
    print(f"Loading HLE dataset from Hugging Face: {dataset}")
    dataset_obj = load_dataset(dataset, split="test").to_dict()
    
    # Convert to list of dicts for async parallelism
    questions = [dict(zip(dataset_obj.keys(), values)) for values in zip(*dataset_obj.values())]
    
    # Filter out questions with images if text_only mode
    if text_only:
        questions = [q for q in questions if not q.get('image')]
        print(f"Text-only mode: filtered to {len(questions)} questions without images")

    # Limit samples if requested
    if max_samples:
        questions = questions[:max_samples]
    
    print(f"Loaded {len(questions)} questions")
    return questions


def format_message(question):
    """Format question into message format for the model."""
    question_text = question['question']
    
    text_content = dict(type="text", text=question_text)
    
    # Add image if present
    if question.get('image'):
        image_content = dict(type="image_url", image_url=dict(url=question['image']))
        content = [text_content, image_content]
    else:
        content = [text_content]
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return messages


async def get_model_prediction_and_judge(model_agent, judge_agent, question, question_idx, max_attempts: int = 5):
    """Get model prediction and immediately judge it for a single question."""
    messages = format_message(question)
    
    # Step 1: Get model prediction
    content = None
    for attempt in range(max_attempts):
        try:
            response = await model_agent.async_completions(messages=messages)
            content = response.content
            assert content is not None, "Model returned None content"
            break
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"\n{max_attempts} prediction attempts failed for question {question['id']}: {e}")
            
    
    # If prediction failed, return a result with empty response (skip judge)
    if content is None:
        return {
            **question,
            'question_idx': question_idx,
            'response': None,
            'judge_response': None,
            'is_correct': False
        }
    
    # Step 2: Judge the answer
    question_text = question["question"]
    correct_answer = question["answer"]
    
    prompt = JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=content
    )
    
    judge_response = None
    for attempt in range(max_attempts):
        try:
            judge_messages = [{"role": "user", "content": prompt}]
            judge_result = await judge_agent.async_completions(messages=judge_messages)
            judge_content = judge_result.content
            
            extracted_answer_match = re.search(r'<extracted_final_answer>(.*?)</extracted_final_answer>', judge_content, re.DOTALL | re.IGNORECASE)
            reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', judge_content, re.DOTALL | re.IGNORECASE)
            correct_match = re.search(r'<correct>(yes|no)</correct>', judge_content, re.DOTALL | re.IGNORECASE)
            confidence_match = re.search(r'<confidence>(\d+)</confidence>', judge_content, re.DOTALL | re.IGNORECASE)
            
            extracted_answer = extracted_answer_match.group(1).strip() if extracted_answer_match else None
            reasoning = reasoning_match.group(1).strip() if reasoning_match else None
            correct = correct_match.group(1).lower() if correct_match else None
            confidence = int(confidence_match.group(1)) if confidence_match else None
            
            assert all(x is not None for x in [correct, reasoning, confidence, extracted_answer]), "Missing required fields in judge response"
            judge_response = dict(
                extracted_answer=extracted_answer,
                reasoning=reasoning,
                correct=correct,
                confidence=confidence
            )
            break
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"\n{max_attempts} judge attempts failed for question {question['id']}: {e}")
    
    # If judge failed, return None
    if judge_response is None:
        return None
    
    # Return complete result with both prediction and judgment
    return {
        **question,
        'question_idx': question_idx,
        'response': content,
        'judge_response': judge_response,
        'is_correct': "yes" in judge_response["correct"]
    }

# Source: https://github.com/hendrycks/outlier-exposure/blob/master/utils/calibration_tools.py
def calib_err(confidence, correct, p='2', beta=100):
    """Calculate calibration error."""
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    
    # Handle case when we have fewer samples than beta
    if len(confidence) < beta:
        # Use all samples as a single bin
        if len(confidence) > 0:
            difference = np.abs(np.nanmean(confidence) - np.nanmean(correct))
            return difference
        return 0.0
    
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)

    return cerr


def compute_metrics(predictions, total_questions, num_failed):
    """Compute accuracy and calibration metrics from predictions."""
    correct = []
    confidence = []
    
    for prediction in predictions:
        if prediction.get('judge_response'):
            judge_response = prediction['judge_response']
            correct.append("yes" in judge_response["correct"])
            confidence.append(judge_response["confidence"])
    
    correct = np.array(correct)
    confidence = np.array(confidence) / 100
    
    num_evaluated = len(correct)
    print(f"Successfully evaluated: {num_evaluated} | Failed: {num_failed} | Total questions: {total_questions}")
    
    # Calculate accuracy for success cases only
    accuracy_success_only = 100 * sum(correct) / num_evaluated if num_evaluated > 0 else 0.0
    # Wald estimator, 95% confidence interval (for success cases)
    confidence_half_width_success = 1.96 * math.sqrt(accuracy_success_only * (100 - accuracy_success_only) / num_evaluated) if num_evaluated > 0 else 0.0
    calibration_error = 100 * calib_err(confidence, correct, p='2', beta=100) if num_evaluated > 0 else 0.0
    
    # Calculate overall accuracy (treating failed as incorrect)
    accuracy_overall = 100 * sum(correct) / total_questions if total_questions > 0 else 0.0
    confidence_half_width_overall = 1.96 * math.sqrt(accuracy_overall * (100 - accuracy_overall) / total_questions) if total_questions > 0 else 0.0
    
    return {
        'accuracy': round(accuracy_overall, 2),
        'confidence_interval': round(confidence_half_width_overall, 2),
        'accuracy_success_only': round(accuracy_success_only, 2),
        'confidence_interval_success_only': round(confidence_half_width_success, 2),
        'calibration_error': round(calibration_error, 2),
        'evaluated_questions': num_evaluated,
        'failed_questions': num_failed,
        'total_questions': total_questions,
    }

async def generate_predictions_and_judge(model_agent,
                                          judge_agent,
                                          dataset: str,
                                          output_file: str,
                                          max_concurrent: int = 10,
                                          text_only: bool = False,
                                          max_samples: int = None,
                                          existing_results: list = None):
    """Generate model predictions and judge them in parallel.

    Results are written incrementally to the JSONL output_file as they complete.
    """

    # Load HLE dataset (filter by text_only at load time)
    questions = load_hle_data(dataset, max_samples=max_samples, text_only=text_only)
    total_questions = len(questions)

    # If not redoing, filter to only questions that need running (missing or empty response)
    existing_by_id = {}
    if existing_results:
        for r in existing_results:
            existing_by_id[r['id']] = r
        # Only skip questions that have a good response (non-None)
        existing_good_ids = {r['id'] for r in existing_results if r.get('response')}
        questions_to_run = [q for q in questions if q['id'] not in existing_good_ids]
        print(f"Skipping {len(existing_good_ids)} completed, rerunning {len(questions_to_run)} (empty: {len(existing_by_id) - len(existing_good_ids)}, missing: {len(questions_to_run) - (len(existing_by_id) - len(existing_good_ids))})")
        questions = questions_to_run

    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    async def predict_and_judge_with_semaphore(question, idx):
        async with semaphore:
            return await get_model_prediction_and_judge(model_agent, judge_agent, question, idx)

    # Process all questions
    print(f"Generating predictions and judging for {len(questions)} questions...")
    tasks = [predict_and_judge_with_semaphore(question, idx) for idx, question in enumerate(questions)]

    # Open the JSONL file in append mode for incremental writes
    output_path = Path(output_file)
    jsonl_file = open(output_path, 'a')
    # Use a lock to serialize writes from concurrent tasks
    write_lock = asyncio.Lock()

    successful_results = []
    failed_results = []
    correct = 0
    num_failed = 0
    confidences = []
    corrects = []
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
    for task in pbar:
        result = await task
        if result is None:
            # This should not happen anymore since we return dicts for empty responses,
            # but handle it defensively
            num_failed += 1
        elif result.get('response') is None:
            # Empty response - model prediction failed
            failed_results.append(result)
            num_failed += 1
        else:
            # Successful result with response and judge
            successful_results.append(result)
            correct += int(result['is_correct'])
            confidences.append(result['judge_response']['confidence'])
            corrects.append(int(result['is_correct']))

        # Write result to JSONL immediately (strip heavy image fields first)
        if result is not None:
            write_result = {k: v for k, v in result.items() if k not in ('image', 'image_preview', 'rationale_image')}
            async with write_lock:
                jsonl_file.write(json.dumps(write_result) + '\n')
                jsonl_file.flush()

        total_processed = len(successful_results) + num_failed
        accuracy = 100 * correct / total_processed if total_processed > 0 else 0.0
        model_cost = model_agent.all_token_usage.cost
        postfix = {
            "acc": f"{accuracy:.1f}%",
            "cost": f"${model_cost:.3f}",
            "failed": num_failed
        }
        if len(confidences) >= 10:
            ece = calib_err(np.array(confidences) / 100, np.array(corrects, dtype=float), beta=10)
            postfix["ce"] = f"{ece:.3f}"
        pbar.set_postfix(postfix)

    jsonl_file.close()

    # Build complete results list by merging existing good results with new results
    if existing_by_id:
        # Start from existing good results
        merged_by_id = {qid: r for qid, r in existing_by_id.items() if r.get('response')}
        # Overwrite/add new successful results
        for r in successful_results:
            merged_by_id[r['id']] = r
        # Collect all successful results for metrics
        all_successful = list(merged_by_id.values())
        # Count failures: total questions minus successful
        total_failed = total_questions - len(all_successful)
    else:
        all_successful = successful_results
        total_failed = num_failed

    # Compute final metrics
    metrics = compute_metrics(all_successful, total_questions, total_failed)

    return all_successful, metrics


def run_eval(model: str,
             output_file: str = None,
             dataset: str = "cais/hle",
             judge_model: str = "gpt-5-mini",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 10,
             text_only: bool = False,
             max_samples: int = None,
             redo: bool = True):
    """
    Run HLE evaluation.

    Args:
        model: Model name from models.yaml (required)
        dataset: HuggingFace dataset identifier
        output_file: Path to output file (required)
        judge_model: Judge model name from models.yaml (default: gpt-5-mini)
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        text_only: If True, filter out questions with images
        max_samples: If set, limit evaluation to first N samples
        redo: If True, rerun all. If False, skip completed questions and only rerun empty/missing
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results from JSONL if not redoing
    existing_results = None
    if not redo and output_path.exists():
        existing_results = []
        with open(output_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    # Skip metrics summary lines (they have 'metrics' key but no 'id' key)
                    if 'id' in entry:
                        existing_results.append(entry)
                except json.JSONDecodeError:
                    continue
        print(f"Loading existing results from {output_path}: {len(existing_results)} entries")

        # When resuming, rewrite the JSONL to only contain existing good results
        # (removes stale empty-response entries that will be retried, and old metrics lines)
        good_results = [r for r in existing_results if r.get('response')]
        with open(output_path, 'w') as f:
            for r in good_results:
                f.write(json.dumps(r) + '\n')
        print(f"Rewrote {output_path} with {len(good_results)} good results for resume")

    model_agent = get_llm_agent_class(**get_agent_config(model, models_config))
    judge_agent = get_llm_agent_class(**get_agent_config(judge_model, models_config))

    # Generate predictions and judge them (results are written incrementally to JSONL)
    results, metrics = asyncio.run(
        generate_predictions_and_judge(
            model_agent=model_agent,
            judge_agent=judge_agent,
            dataset=dataset,
            output_file=output_file,
            max_concurrent=max_concurrent,
            text_only=text_only,
            max_samples=max_samples,
            existing_results=existing_results
        )
    )

    print("\n=== HLE Results ===")
    print(f"Dataset: {dataset}")
    print(f"Overall Accuracy: {metrics['accuracy']}% ± {metrics['confidence_interval']}% (treating failed as incorrect)")
    print(f"Success-only Accuracy: {metrics['accuracy_success_only']}% ± {metrics['confidence_interval_success_only']}% (only successful evaluations)")
    print(f"Calibration Error: {metrics['calibration_error']}")
    print(f"Evaluated: {metrics['evaluated_questions']} questions")
    print(f"Failed: {metrics['failed_questions']} questions")
    print(f"Total: {metrics['total_questions']} questions")

    print("\n===== Token Usage =====")
    print(f"Model: {model_agent.all_token_usage} | Max: {model_agent.max_token_usage}")
    print(f"Judge: {judge_agent.all_token_usage} | Max: {judge_agent.max_token_usage}")
    print(f"Total Cost: ${model_agent.all_token_usage.cost + judge_agent.all_token_usage.cost:.4f}")

    # Append a metrics summary line to the JSONL
    metrics_line = {
        '_type': 'metrics_summary',
        'model': model,
        'judge_model': judge_model,
        'dataset': dataset,
        'metrics': metrics,
    }
    with open(output_path, 'a') as f:
        f.write(json.dumps(metrics_line) + '\n')

    print(f"\nResults saved to {output_path} (JSONL format, {len(results)} result lines + metrics summary)")


if __name__ == '__main__':
    fire.Fire(run_eval)



