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


async def get_model_prediction(agent, question, max_attempts: int = 3):
    """Get model prediction for a single question."""
    messages = format_message(question)
    
    # Try to get response with retries
    content = None
    for attempt in range(max_attempts):
        try:
            response = await agent.async_completions(messages=messages)
            content = response.content
            assert content is not None, "Model returned None content"
            break
            
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"{max_attempts} attempts failed for question {question['id']} : {e}")
                
    return dict(**question, response=content)


async def judge_answer(judge_agent, question, max_attempts: int = 3):
    """Judge if the model's answer is correct using XML parsing."""
    question_text = question["question"]
    correct_answer = question["answer"]
    response = question["response"]
    
    prompt = JUDGE_PROMPT.format(
        question=question_text,
        correct_answer=correct_answer,
        response=response
    )
    
    judge_response = None
    for attempt in range(max_attempts):
        try:
            judge_messages = [{"role": "user", "content": prompt}]
            response = await judge_agent.async_completions(messages=judge_messages)
            judge_content = response.content
            
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
                print(f"{max_attempts} judge attempts failed for question {question['id']}: {e}")
    
    # Update question with new judge response (overrides existing if present)
    result = dict(**question)
    result['judge_response'] = judge_response
    return result

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


def compute_metrics(predictions, total_questions):
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
    
    print(f"Available predictions: {len(correct)} | Total questions: {total_questions}")
    
    accuracy = 100 * sum(correct) / total_questions
    # Wald estimator, 95% confidence interval
    confidence_half_width = 1.96 * math.sqrt(accuracy * (100 - accuracy) / total_questions)
    calibration_error = 100 * calib_err(confidence, correct, p='2', beta=100)
    
    return {
        'accuracy': round(accuracy, 2),
        'confidence_interval': round(confidence_half_width, 2),
        'calibration_error': round(calibration_error, 2),
        'evaluated_questions': len(correct),
    }

async def generate_predictions(agent,
                                dataset: str,
                                max_concurrent: int = 10,
                                text_only: bool = False,
                                max_samples: int = None):
    """Generate model predictions without judging."""
    
    # Load HLE dataset (filter by text_only at load time)
    questions = load_hle_data(dataset, max_samples=max_samples, text_only=text_only)
    
    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def predict_with_semaphore(question):
        async with semaphore:
            return await get_model_prediction(agent, question)
    
    # Process all questions
    print(f"Generating predictions for {len(questions)} questions...")
    tasks = [predict_with_semaphore(question) for question in questions]
    
    results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating predictions"):
        result = await task
        if result:
            results.append(result)
    
    return results


async def judge_predictions(judge_agent,
                             questions: list,
                             max_concurrent: int = 128):
    """Judge existing predictions."""    
    # Filter to questions that have predictions
    print(f"Judging {len(questions)} predictions...")
    
    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def judge_with_semaphore(question):
        async with semaphore:        
            judge_result = await judge_answer(judge_agent, question)
            return judge_result  # Already contains all question fields
    
    # Process all questions
    tasks = [judge_with_semaphore(question) for question in questions]
    
    judged_results = []
    for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Judging predictions"):
        result = await task
        judged_results.append(result)
    
    # Compute metrics
    metrics = compute_metrics(judged_results, len(questions))
    
    return judged_results, metrics


def run_eval(model: str,
             output_file: str = None,
             dataset: str = "cais/hle",
             judge_model: str = "gpt-5-mini",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 10,
             text_only: bool = False,
             max_samples: int = None,
             judge_only: bool = False):
    """
    Run HLE evaluation.
    
    Args:
        model: Model name from models.yaml (required unless judge_only=True)
        dataset: HuggingFace dataset identifier
        output_file: Path to output file (required)
        judge: Judge model name from models.yaml (default: gpt-5-mini)
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        text_only: If True, filter out questions with images
        max_samples: If set, limit evaluation to first N samples
        judge_only: If True, load predictions from output_file and only run judge
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model_agent = get_llm_agent_class(**get_agent_config(model, models_config))
    judge_agent = get_llm_agent_class(**get_agent_config(judge_model, models_config))
    
    if not judge_only:
        # Step 1: Generate predictions
        predictions = asyncio.run(
            generate_predictions(
                agent=model_agent,
                dataset=dataset,
                max_concurrent=max_concurrent,
                text_only=text_only,
                max_samples=max_samples
            )
        )
    else:
        # Judge-only mode: load existing predictions and judge them
        if not output_path.exists():
            raise ValueError(f"Cannot run judge_only mode: {output_file} does not exist")
        
        print(f"===>Running in judge-only mode. Loading predictions from: {output_path}")
        
        with open(output_path, 'r') as f:
            saved_data = json.load(f)
        predictions = saved_data

    # Step 2: Judge predictions
    results, metrics = asyncio.run(
        judge_predictions(
            judge_agent=judge_agent,
            questions=predictions,
            max_concurrent=max_concurrent
        )
    )

    print("\n=== HLE Judge Results ===")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {metrics['accuracy']}% ± {metrics['confidence_interval']}%")
    print(f"Calibration Error: {metrics['calibration_error']}")
    print(f"Evaluated: {metrics['evaluated_questions']} questions")
    
    print("\n===== Token Usage =====")
    print(f"Model: {model_agent.all_token_usage} | Max: {model_agent.max_token_usage}")
    print(f"Judge: {judge_agent.all_token_usage} | Max: {judge_agent.max_token_usage}")
    
    # Save results
    print(f"\nSaving results to {output_path}")
    
    # Remove heavy fields that we don't need to save
    for result in results:
        result.pop("image", None)
        result.pop("image_preview", None)
        result.pop("rationale_image", None)

    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved successfully!")


if __name__ == '__main__':
    fire.Fire(run_eval)



