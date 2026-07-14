#!/usr/bin/env python3
"""
ERQA (Embodied Reasoning QA) Evaluation Script
Evaluates vision-language models on embodied reasoning with multimodal questions.

Mirrors the HLE / EnigmaEval refactor:
  * Incremental JSONL output (one flushed record per item)
  * Every item is saved, including parse failures (extracted_answer = None),
    so no data is ever silently dropped.
  * redo=True/False resume with HLE semantics (redo=False skips items that
    already have a valid A-D answer, reruns the missing / null ones).
  * A \text{}-robust answer extractor (balanced-brace + LaTeX-wrapper strip),
    reused from mindcube, so \boxed{\text{A}} extracts to "A".
  * Final accuracy is read back from the JSONL and printed as "Accuracy: X%".
"""

import asyncio
import os
import json
import re
import shutil
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
    """Load ERQA dataset from Hugging Face.

    The shuffle(seed=42) + triple is deterministic, so the enumerate index
    (``example_idx``) is a stable id across runs and can be used to resume.
    """
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


# =============== ANSWER EXTRACTION (\text{}-robust) ===============
def _last_boxed_content(text: str):
    """Return the content of the LAST ``\\boxed{...}`` in ``text`` using
    balanced-brace matching, or None if there is no ``\\boxed{``.

    The old regex ``r'\\boxed\{([A-D])\}'`` was STRICT: a nested wrapper like
    ``\\boxed{\\text{A}}`` did not match at all, so the item failed to parse,
    retried 3x, and was dropped. Balancing braces here captures the full inner
    content (``\\text{A}``) so the letter can be recovered downstream.
    """
    matches = list(re.finditer(r'\\boxed\s*\{', text, re.IGNORECASE))
    if not matches:
        return None
    open_brace = matches[-1].end() - 1  # index of the opening '{'
    depth = 0
    for j in range(open_brace, len(text)):
        if text[j] == '{':
            depth += 1
        elif text[j] == '}':
            depth -= 1
            if depth == 0:
                return text[open_brace + 1:j]
    return text[open_brace + 1:]  # unbalanced: take the remainder


def _extract_choice_letter(text: str):
    """Pull the final standalone A-D choice letter out of a (possibly
    LaTeX-wrapped) string. Returns the uppercase letter, or None.

    Handles ``\\text{A}``, ``\\mathrm{A}``, ``\\textbf{A}``, stray ``$``,
    surrounding whitespace, and trailing punctuation. Only single, isolated
    letters count, so a stray letter inside a word (e.g. the "A" in "Answer")
    is never grabbed.
    """
    # Strip LaTeX command wrappers, keeping their braced argument content.
    cleaned = re.sub(
        r'\\(?:text|mathrm|textbf|mathbf|textit|mathit|boxed|rm|bf|it)\s*',
        '', text)
    # Drop braces / dollar signs / stray backslashes so only the letter remains.
    for ch in ('{', '}', '$', '\\'):
        cleaned = cleaned.replace(ch, ' ')
    # Take the LAST standalone A-D letter (not part of a longer word).
    letters = re.findall(r'(?<![A-Za-z])([A-Da-d])(?![A-Za-z])', cleaned)
    if not letters:
        return None
    return letters[-1].upper()


def parse_answer(response_text: str, ground_truth: str):
    """Parse and evaluate an answer from a model response.

    Returns a dict {'extracted_answer', 'is_correct'} on success, or None if no
    A-D answer could be recovered from the last ``\\boxed{...}`` block.
    """
    if not isinstance(response_text, str):
        return None

    # Extract from the LAST boxed block, handling LaTeX-wrapped answers such
    # as \boxed{\text{A}}. See _last_boxed_content for the \text{} fix.
    boxed_content = _last_boxed_content(response_text)
    if boxed_content is None:
        return None  # No boxed answer found

    extracted_answer = _extract_choice_letter(boxed_content)
    if extracted_answer is None:
        return None  # Boxed block had no A-D choice letter

    is_correct = extracted_answer == ground_truth.strip().upper()

    return {
        'extracted_answer': extracted_answer,
        'is_correct': is_correct
    }


def make_record(example, example_idx, response, parse_result):
    """Build a lightweight JSONL record for one item (no heavy image fields)."""
    return {
        'example_idx': example_idx,
        'question': example.get('question'),
        'answer': example.get('answer'),
        'question_type': example.get('question_type'),
        'visual_indices': example.get('visual_indices'),
        'num_images': len(example.get('images_base64', [])),
        'response': response,
        'extracted_answer': parse_result['extracted_answer'] if parse_result else None,
        'is_correct': parse_result['is_correct'] if parse_result else False,
    }


def read_jsonl_records(path: str):
    """Read item records from a JSONL file (skips the metrics summary line)."""
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
            if 'example_idx' in entry:
                records.append(entry)
    return records


def compute_metrics(records):
    """Compute accuracy metrics from JSONL records.

    Primary accuracy is computed over items that produced a valid A-D answer
    (success-only), matching the historical ERQA number. Items with a null
    answer (parse / API failure) are reported separately.
    """
    total = len(records)
    valid = [r for r in records if r.get('extracted_answer') is not None]
    num_valid = len(valid)
    num_null = total - num_valid
    correct = sum(1 for r in valid if r.get('is_correct'))

    # Group by question type (valid answers only)
    type_stats = {}
    for r in valid:
        q_type = r.get('question_type', 'unknown')
        if q_type not in type_stats:
            type_stats[q_type] = {'correct': 0, 'total': 0}
        type_stats[q_type]['total'] += 1
        type_stats[q_type]['correct'] += int(bool(r.get('is_correct')))

    return {
        'accuracy': round(100 * correct / num_valid, 2) if num_valid > 0 else 0,
        'accuracy_overall': round(100 * correct / total, 2) if total > 0 else 0,
        'correct': correct,
        'evaluated_questions': num_valid,
        'null_questions': num_null,
        'total_records': total,
        'type_accuracy': {k: round(100 * v['correct'] / v['total'], 2)
                          for k, v in type_stats.items()},
    }


async def get_model_prediction(agent, example, example_idx, max_attempts: int = 3):
    """Get a model prediction for a single example.

    Always returns a record (never None): the raw response is saved even when
    no answer can be parsed, so nothing is dropped. On repeated parse failure
    the last non-empty response is kept with extracted_answer=None; on API
    failure response is None. Both are retryable on a redo=False resume.
    """
    messages = format_message(example)

    last_content = None
    for attempt in range(max_attempts):
        try:
            response = await agent.async_completions(messages=messages)
            content = response.content or response.reasoning_content
            assert content, "Model returned empty content and reasoning_content"
            last_content = content

            parse_result = parse_answer(content, example['answer'])
            if parse_result is not None:
                return make_record(example, example_idx, content, parse_result)
            # Parseable answer not found: retry (response is still retained)
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"{max_attempts} attempts failed for example {example_idx}: {e}")

    # All attempts exhausted: save whatever response we got (may be None), with
    # a null answer so this item is picked up again on a redo=False resume.
    return make_record(example, example_idx, last_content, None)


async def generate_predictions(agent,
                               dataset: str,
                               output_file: str,
                               max_concurrent: int = 10,
                               max_samples: int = None,
                               existing_results: list = None):
    """Generate model predictions for the ERQA dataset.

    Results are written incrementally to the JSONL output_file as they complete,
    so a killed job keeps everything done so far.
    """

    # Load ERQA dataset
    examples = load_erqa_data(dataset, max_samples=max_samples)
    total_examples = len(examples)

    # Resume: skip items that already have a valid A-D answer
    existing_by_id = {}
    if existing_results:
        for r in existing_results:
            existing_by_id[r['example_idx']] = r
        good_ids = {r['example_idx'] for r in existing_results
                    if r.get('extracted_answer') is not None}
        to_run = [(idx, ex) for idx, ex in enumerate(examples) if idx not in good_ids]
        null_count = len(existing_by_id) - len(good_ids)
        print(f"Skipping {len(good_ids)} completed, rerunning {len(to_run)} "
              f"(null/failed: {null_count}, missing: {len(to_run) - null_count})")
    else:
        to_run = list(enumerate(examples))

    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    async def predict_with_semaphore(example, idx):
        async with semaphore:
            return await get_model_prediction(agent, example, idx)

    print(f"Generating predictions for {len(to_run)} examples...")
    tasks = [predict_with_semaphore(example, idx) for idx, example in to_run]

    # Open the JSONL file in append mode for incremental writes
    output_path = Path(output_file)
    jsonl_file = open(output_path, 'a')
    write_lock = asyncio.Lock()

    correct = 0
    processed = 0
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
    for task in pbar:
        result = await task

        # Write result to JSONL immediately (retryable if answer is null)
        async with write_lock:
            jsonl_file.write(json.dumps(result) + '\n')
            jsonl_file.flush()

        processed += 1
        correct += int(bool(result.get('is_correct')))
        accuracy = 100 * correct / processed if processed > 0 else 0.0
        cost = agent.all_token_usage.cost
        pbar.set_postfix({"acc": f"{accuracy:.1f}%", "cost": f"${cost:.3f}"})

    jsonl_file.close()

    # Read back the JSONL to build the authoritative merged result set
    # (dedupe by example_idx, last write wins so retried rows overwrite stale ones)
    merged_by_id = {}
    for r in read_jsonl_records(output_file):
        merged_by_id[r['example_idx']] = r
    all_results = list(merged_by_id.values())

    return all_results, total_examples


def run_eval(model: str,
             output_file: str = None,
             dataset: str = "justinphan3110/erqa",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 10,
             max_samples: int = None,
             redo: bool = True):
    """
    Run ERQA evaluation.

    Args:
        model: Model name from models.yaml
        output_file: Path to the JSONL output file (required)
        dataset: HuggingFace dataset identifier
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        max_samples: If set, limit evaluation to first N samples
        redo: If True (default), rerun everything fresh. If False, resume:
              skip items with a valid A-D answer and only rerun missing/null.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results from JSONL if resuming
    existing_results = None
    if redo:
        # Fresh run: start from an empty JSONL so we don't append to stale rows
        if output_path.exists():
            output_path.unlink()
    else:
        if output_path.exists():
            existing_results = read_jsonl_records(output_file)
            print(f"Loading existing results from {output_path}: {len(existing_results)} entries")

            # GUARD (data-destruction landmine): a NON-empty file that parses to
            # ZERO records is malformed (e.g. a pretty-printed JSON array, not
            # JSONL). Do NOT truncate + fully rerun it — that silently wipes real
            # data. Back it up and abort so a human can recover/convert it.
            if output_path.stat().st_size > 0 and len(existing_results) == 0:
                corrupt_bak = output_path.with_suffix(output_path.suffix + '.corrupt.bak')
                shutil.copy2(output_path, corrupt_bak)
                raise RuntimeError(
                    f"{output_path} is non-empty ({output_path.stat().st_size} bytes) but "
                    f"read_jsonl_records parsed 0 records — refusing to overwrite/rerun. "
                    f"Backed up to {corrupt_bak}. Inspect/convert it before resuming."
                )

            # Rewrite the JSONL RETAINING every row (dedupe by example_idx, last
            # write wins). Incomplete rows (null answer, but saved response) are
            # kept on disk so a crash mid-resume never loses their saved data;
            # the read-back merge dedupes retried rows over these stale ones.
            merged_by_id = {}
            for r in existing_results:
                merged_by_id[r['example_idx']] = r
            with open(output_path, 'w') as f:
                for r in merged_by_id.values():
                    f.write(json.dumps(r) + '\n')
            good = sum(1 for r in merged_by_id.values() if r.get('extracted_answer') is not None)
            print(f"Rewrote {output_path} with {len(merged_by_id)} rows "
                  f"({good} complete, {len(merged_by_id) - good} incomplete/retryable) for resume")

    model_agent = get_llm_agent_class(**get_agent_config(model, models_config))

    # Generate predictions (results are written incrementally to JSONL)
    results, total_examples = asyncio.run(
        generate_predictions(
            agent=model_agent,
            dataset=dataset,
            output_file=output_file,
            max_concurrent=max_concurrent,
            max_samples=max_samples,
            existing_results=existing_results,
        )
    )

    # Compute metrics from the authoritative merged JSONL records
    metrics = compute_metrics(results)

    print("\n=== ERQA Results ===")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {metrics['accuracy']}%")
    print(f"Overall accuracy (null as incorrect): {metrics['accuracy_overall']}% "
          f"({metrics['correct']}/{metrics['total_records']})")
    print(f"Evaluated (valid answer): {metrics['evaluated_questions']}")
    print(f"Null / failed: {metrics['null_questions']}")
    print(f"Total records: {metrics['total_records']} (dataset size: {total_examples})")
    print("\nBy question type:")
    for q_type, acc in metrics['type_accuracy'].items():
        print(f"  {q_type}: {acc}%")

    print("\n===== Token Usage =====")
    print(f"Model: {model_agent.all_token_usage} | Max: {model_agent.max_token_usage}")

    # Append a metrics summary line to the JSONL (authoritative dashboard number)
    metrics_line = {
        '_type': 'metrics_summary',
        'model': model,
        'dataset': dataset,
        'metrics': metrics,
    }
    with open(output_path, 'a') as f:
        f.write(json.dumps(metrics_line) + '\n')

    print(f"\nResults saved to {output_path} (JSONL format, {len(results)} result lines + metrics summary)")


if __name__ == '__main__':
    fire.Fire(run_eval)
