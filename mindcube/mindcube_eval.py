#!/usr/bin/env python3
"""
MindCube-Tiny Evaluation Script

Mirrors the IntPhys2 / SpatialViz / ERQA harness pattern:
  * incremental JSONL output (one record per item, flushed as it completes)
  * redo=True (fresh) / redo=False (resume) semantics
  * the raw response is saved for EVERY item, INCLUDING parse-failures, so a
    later re-parse or resume never loses data (save-on-failure)
  * nulls-as-incorrect: accuracy denominator is the FULL dataset, so parse /
    API failures count as wrong instead of being dropped from the denominator
  * a \\text{}-robust boxed-answer extractor (balanced-brace + LaTeX-wrapper)
  * a data-destruction guard: a non-empty output file that parses to 0 JSONL
    records is never silently truncated + rerun
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
SYSTEM_PROMPT = """You are evaluating spatial mental modeling questions with multimodal content. You will be shown images and text that form questions about spatial reasoning, object relationships, and cognitive mapping. Please analyze the content carefully and provide your response following the required format.

Your final choice should be boxed in the following format:
$\\boxed{choice}$

For example: $\\boxed{A}$"""


def load_mindcube_data(dataset: str, max_samples: int = None):
    """Load MindCube dataset from Hugging Face.

    The dataset load order is deterministic (no shuffle), so the per-item ``id``
    field is a stable key across runs and can be used to resume + to match
    migrated survivors converted from the legacy ``.json`` files.
    """
    print(f"Loading MindCube dataset from Hugging Face: {dataset}")

    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables")

    dataset_obj = load_dataset(dataset, split="train", token=hf_token)
    examples = [dict(example) for example in dataset_obj]

    # Limit samples if requested
    if max_samples:
        examples = examples[:max_samples]

    print(f"Loaded {len(examples)} examples")
    return examples


def format_message(example):
    """Format example into message format for the model."""
    question = example['question']
    base64_images = example['images_base64']

    # Build multimodal content (images first, then question)
    content = []

    for img_b64 in base64_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f'data:image/png;base64,{img_b64}',
                "detail": "high"
            }
        })

    content.append({"type": "text", "text": question})

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content}
    ]
    return messages


# =============== ANSWER EXTRACTION (\text{}-robust) ===============
def _last_boxed_content(text: str):
    """Return the content of the LAST ``\\boxed{...}`` in ``text`` using
    balanced-brace matching, or None if there is no ``\\boxed{``.

    The old regex ``r'\\boxed\{([^}]+)\}'`` stopped at the FIRST ``}``, so a
    nested wrapper like ``\\boxed{\\text{A}}`` captured only ``\\text{A``.
    Balancing braces here captures the full inner content (``\\text{A}``) so
    the answer letter can be recovered downstream.
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
    """Pull the final standalone A-E choice letter out of a (possibly
    LaTeX-wrapped) string. Returns the uppercase letter, or None.

    Handles ``\\text{A}``, ``\\mathrm{A}``, ``\\textbf{A}``, stray ``$``,
    surrounding whitespace, and trailing punctuation. Only single, isolated
    letters count, so a stray letter inside a word (e.g. the "A" in "Answer")
    is never grabbed.
    """
    if not isinstance(text, str):
        return None
    # Strip LaTeX command wrappers, keeping their braced argument content.
    cleaned = re.sub(
        r'\\(?:text|mathrm|textbf|mathbf|textit|mathit|boxed|rm|bf|it)\s*',
        '', text)
    # Drop braces / dollar signs / stray backslashes so only the letter remains.
    for ch in ('{', '}', '$', '\\'):
        cleaned = cleaned.replace(ch, ' ')
    # Take the LAST standalone A-E letter (not part of a longer word).
    letters = re.findall(r'(?<![A-Za-z])([A-Ea-e])(?![A-Za-z])', cleaned)
    if not letters:
        return None
    return letters[-1].upper()


def parse_answer(response_text: str, ground_truth: str):
    """Parse and evaluate answer from a response.

    Extracts from the LAST boxed block, handling LaTeX-wrapped answers such as
    ``\\boxed{\\text{A}}``. Returns None only when no A-E choice letter can be
    recovered from a boxed block (so the raw response is still saved upstream).
    """
    if not isinstance(response_text, str):
        return None

    boxed_content = _last_boxed_content(response_text)
    if boxed_content is None:
        return None  # No boxed answer found

    extracted_answer = _extract_choice_letter(boxed_content)
    if extracted_answer is None:
        return None  # Boxed block had no A-E choice letter

    is_correct = extracted_answer == ground_truth.strip().upper()

    return {
        'extracted_answer': extracted_answer,
        'is_correct': is_correct
    }


# =============== JSONL I/O ===============
def read_jsonl_records(path: str) -> list:
    """Read per-item result records from a JSONL file.

    Skips blank lines and the trailing ``_type == 'metrics_summary'`` line.
    """
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
            if 'id' in entry:
                records.append(entry)
    return records


def compute_metrics(results, total_examples):
    """Compute accuracy metrics.

    ``total_examples`` is the full dataset size (denominator). Items that were
    parse-fails / API-fails / missing count as incorrect (they are saved with
    is_correct=False / are simply absent from ``results``).
    """
    total = total_examples if total_examples else len(results)
    correct = sum(int(r.get('is_correct', False)) for r in results)
    evaluated = sum(1 for r in results if r.get('extracted_answer') is not None)

    # Group by MindCube categories (around, rotation, among)
    category_stats = {}
    for r in results:
        setting = r.get('setting', 'other')
        if setting not in category_stats:
            category_stats[setting] = {'correct': 0, 'total': 0}
        category_stats[setting]['total'] += 1
        category_stats[setting]['correct'] += int(r.get('is_correct', False))

    return {
        'accuracy': round(100 * correct / total, 2) if total > 0 else 0,
        'correct': correct,
        'total': total,
        'evaluated': evaluated,
        'failed': total - evaluated,
        'stored': len(results),
        'category_accuracy': {k: round(100 * v['correct'] / v['total'], 2)
                             for k, v in category_stats.items() if v['total'] > 0},
        'category_counts': {k: f"{v['correct']}/{v['total']}"
                           for k, v in category_stats.items() if v['total'] > 0}
    }


async def get_model_prediction(agent, example, example_idx, max_attempts: int = 3):
    """Get the model prediction for a single example.

    ALWAYS returns a record (never None). The raw response is saved even when the
    answer cannot be parsed, so nothing is lost:
      * API/empty-content failure -> response=None, extracted_answer=None (retryable)
      * parse failure              -> response=<text>, extracted_answer=None (retryable + recoverable)
      * success                    -> response=<text>, extracted_answer='A'..'E'
    """
    messages = format_message(example)

    # Retry only on API errors / empty content (not on parse failures).
    content = None
    for attempt in range(max_attempts):
        try:
            response = await agent.async_completions(messages=messages)
            content = response.content or response.reasoning_content
            assert content, "Model returned empty content and reasoning_content"
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"{max_attempts} attempts failed for example {example_idx}: {e}")

    # Metadata common to every record (drop the heavy images field).
    base = {k: v for k, v in example.items() if k != 'images_base64'}
    base['example_idx'] = example_idx
    base['num_images'] = len(example['images_base64'])

    if content is None:
        # API failure: retryable on resume.
        return {
            **base,
            'response': None,
            'extracted_answer': None,
            'is_correct': False,
        }

    parse_result = parse_answer(content, example['answer'])
    if parse_result is None:
        # Parse failure: SAVE the raw response so it is never lost. Marked
        # retryable (extracted_answer=None) so a redo=False rerun revisits it.
        return {
            **base,
            'response': content,
            'extracted_answer': None,
            'is_correct': False,
        }

    return {
        **base,
        'response': content,
        'extracted_answer': parse_result['extracted_answer'],
        'is_correct': parse_result['is_correct'],
    }


async def generate_predictions(agent,
                               dataset: str,
                               output_file: str,
                               max_concurrent: int = 10,
                               max_samples: int = None,
                               existing_results: list = None):
    """Generate model predictions for the MindCube dataset.

    Results are written incrementally to the JSONL ``output_file`` as they
    complete (one flushed line per item, including parse failures), so a killed
    job keeps everything done so far.
    """
    # Load MindCube dataset
    examples = load_mindcube_data(dataset, max_samples=max_samples)
    total_examples = len(examples)

    # Resume: skip items that already have a valid parsed answer.
    existing_by_id = {}
    if existing_results:
        for r in existing_results:
            existing_by_id[r['id']] = r
        good_ids = {r['id'] for r in existing_results
                    if r.get('extracted_answer') is not None}
        examples_to_run = [ex for ex in examples if ex['id'] not in good_ids]
        empty = len(existing_by_id) - len(good_ids)
        print(
            f"Skipping {len(good_ids)} completed, rerunning {len(examples_to_run)} "
            f"(empty/parse-fail: {empty}, missing: {len(examples_to_run) - empty})"
        )
        examples = examples_to_run

    if not examples:
        print("No examples to process")

    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    async def predict_with_semaphore(example, idx):
        async with semaphore:
            return await get_model_prediction(agent, example, idx)

    print(f"Generating predictions for {len(examples)} examples...")
    tasks = [predict_with_semaphore(example, idx) for idx, example in enumerate(examples)]

    # Open the JSONL file in append mode for incremental writes.
    output_path = Path(output_file)
    jsonl_file = open(output_path, 'a')
    write_lock = asyncio.Lock()

    correct = 0
    processed = 0
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
    for task in pbar:
        result = await task

        # Write EVERY result immediately (including parse-fails / API-fails).
        async with write_lock:
            jsonl_file.write(json.dumps(result) + '\n')
            jsonl_file.flush()

        processed += 1
        correct += int(result['is_correct'])
        accuracy = 100 * correct / processed if processed > 0 else 0.0
        cost = agent.all_token_usage.cost
        pbar.set_postfix({"acc": f"{accuracy:.1f}%", "cost": f"${cost:.3f}"})

    jsonl_file.close()

    # Read back the JSONL to build the authoritative merged result set
    # (dedupe by id, last write wins so retried rows overwrite stale ones).
    merged_by_id = {}
    for r in read_jsonl_records(output_file):
        merged_by_id[r['id']] = r
    all_results = list(merged_by_id.values())

    return all_results, total_examples


def run_eval(model: str,
             output_file: str = None,
             dataset: str = "justinphan3110/mindcube",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 10,
             max_samples: int = None,
             redo: bool = True):
    """
    Run MindCube evaluation.

    Args:
        model: Model name from models.yaml
        output_file: Path to output JSONL file (required)
        dataset: HuggingFace dataset identifier
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        max_samples: If set, limit evaluation to first N samples
        redo: If True (default), rerun all. If False, resume: skip items with a
              valid parsed answer and only rerun missing/null-answer items.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results from JSONL for resume (redo=False).
    existing_results = None
    if redo:
        # Fresh run: start from an empty JSONL so we don't append to stale rows.
        if output_path.exists():
            output_path.unlink()
    else:
        if output_path.exists():
            existing_results = read_jsonl_records(str(output_path))
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

            # Rewrite the JSONL RETAINING every row (dedupe by id, last write
            # wins). Incomplete rows (null answer, but saved response) are kept
            # on disk so a crash mid-resume never loses their saved data; the
            # read-back merge dedupes retried rows over these stale ones.
            merged_by_id = {}
            for r in existing_results:
                merged_by_id[r['id']] = r
            with open(output_path, 'w') as f:
                for r in merged_by_id.values():
                    f.write(json.dumps(r) + '\n')
            good = sum(1 for r in merged_by_id.values() if r.get('extracted_answer') is not None)
            print(f"Rewrote {output_path} with {len(merged_by_id)} rows "
                  f"({good} complete, {len(merged_by_id) - good} incomplete/retryable) for resume")

    model_agent = get_llm_agent_class(**get_agent_config(model, models_config))

    # Generate predictions (results are written incrementally to JSONL).
    results, total_examples = asyncio.run(
        generate_predictions(
            agent=model_agent,
            dataset=dataset,
            output_file=str(output_path),
            max_concurrent=max_concurrent,
            max_samples=max_samples,
            existing_results=existing_results,
        )
    )

    # Compute metrics (denominator = full dataset; parse-fails/missing = incorrect).
    metrics = compute_metrics(results, total_examples)

    print("\n=== MindCube Results ===")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {metrics['accuracy']}% ({metrics['correct']}/{metrics['total']})")
    print(f"Evaluated: {metrics['evaluated']} | Failed/null: {metrics['failed']} | "
          f"Stored: {metrics['stored']} | Total dataset: {metrics['total']}")
    print("\nBy category:")
    for category, acc in metrics['category_accuracy'].items():
        counts = metrics['category_counts'][category]
        print(f"  {category.capitalize()}: {acc}% ({counts})")

    print("\n===== Token Usage =====")
    print(f"Model: {model_agent.all_token_usage} | Max: {model_agent.max_token_usage}")

    # Append a metrics summary line to the JSONL (authoritative dashboard number).
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
