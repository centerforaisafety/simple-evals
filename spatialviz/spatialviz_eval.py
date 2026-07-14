#!/usr/bin/env python3
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


def make_example_id(example) -> str:
    """Build a stable, shuffle-independent id for an example.

    The composite ``Category/Task/Level/Image_id`` is unique across the whole
    dataset (verified) and does not depend on the shuffle seed, so it survives
    resume runs and legacy-JSON migration alike.
    """
    return f"{example['Category']}/{example['Task']}/{example['Level']}/{example['Image_id']}"


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


def _last_boxed_content(text: str):
    """Return the content of the LAST ``\\boxed{...}`` in ``text`` using
    balanced-brace matching, or None if there is no ``\\boxed{``.

    A strict regex like ``r'\\boxed\{([A-D])\}'`` fails to match a nested
    wrapper such as ``\\boxed{\\text{A}}`` at all, which previously forced a
    retry and eventually dropped the item. Balancing braces here captures the
    full inner content (``\\text{A}``) so the letter can be recovered.
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
    """Parse and evaluate the answer from a response.

    Extracts from the LAST boxed block, handling LaTeX-wrapped answers such as
    ``\\boxed{\\text{A}}``. Returns None only when no A-D choice letter can be
    recovered from a boxed block (so the raw response is still saved upstream).
    """
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


def read_jsonl_records(path: str) -> list:
    """Read per-item result records from a JSONL file (skips metrics summary)."""
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
    """Compute accuracy metrics from results.

    ``total_examples`` is the full dataset size; items with no valid parsed
    answer count as incorrect in the overall accuracy (HLE semantics).
    """
    total = total_examples if total_examples else len(results)
    correct = sum(int(r.get('is_correct', False)) for r in results)
    evaluated = sum(1 for r in results if r.get('extracted_answer') is not None)

    # Group by category
    category_stats = {}
    for r in results:
        category = r['Category']
        if category not in category_stats:
            category_stats[category] = {'correct': 0, 'total': 0}
        category_stats[category]['total'] += 1
        category_stats[category]['correct'] += int(r.get('is_correct', False))

    # Group by task
    task_stats = {}
    for r in results:
        task = r['Task']
        if task not in task_stats:
            task_stats[task] = {'correct': 0, 'total': 0}
        task_stats[task]['total'] += 1
        task_stats[task]['correct'] += int(r.get('is_correct', False))

    return {
        'accuracy': round(100 * correct / total, 2) if total > 0 else 0,
        'correct': correct,
        'total': total,
        'evaluated': evaluated,
        'failed': total - evaluated,
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
    """Get model prediction for a single example.

    ALWAYS returns a result dict so the raw response is never lost: on API
    failure ``response`` is None; on a parse failure ``extracted_answer`` is
    None but the raw ``response`` is preserved. Both cases are retryable on a
    later ``--redo=False`` resume (they have no valid parsed answer).
    """
    example_id = make_example_id(example)

    def build_result(content, parse_result):
        extracted = parse_result['extracted_answer'] if parse_result else None
        is_correct = parse_result['is_correct'] if parse_result else False
        return {
            **example,
            'id': example_id,
            'example_idx': example_idx,
            'response': content,
            'extracted_answer': extracted,
            'is_correct': is_correct,
        }

    # Retry only on API/empty-response errors, NOT on parse failures.
    content = None
    for attempt in range(max_attempts):
        try:
            response = await agent.async_completions(messages=format_message(example))
            content = response.content or response.reasoning_content
            assert content, "Model returned empty content and reasoning_content"
            break
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"{max_attempts} attempts failed for example {example_idx} ({example_id}): {e}")
                return build_result(None, None)

    # Parse the answer; keep the raw response even if parsing fails.
    parse_result = parse_answer(content, example['Answer'])
    return build_result(content, parse_result)


async def generate_predictions(agent,
                                dataset: str,
                                output_file: str,
                                max_concurrent: int = 10,
                                max_samples: int = None,
                                existing_results: list = None):
    """Generate model predictions for the SpatialViz dataset.

    Results are written incrementally to the JSONL ``output_file`` as they
    complete (one flushed line per item, including parse failures), so a killed
    job keeps everything done so far.
    """

    # Load SpatialViz dataset
    examples = load_spatialviz_data(dataset, max_samples=max_samples)
    total_examples = len(examples)

    # If resuming, skip examples that already have a valid parsed answer
    existing_by_id = {}
    if existing_results:
        for r in existing_results:
            existing_by_id[r['id']] = r
        good_ids = {r['id'] for r in existing_results if r.get('extracted_answer') is not None}
        examples_to_run = [ex for ex in examples if make_example_id(ex) not in good_ids]
        empty = len(existing_by_id) - len(good_ids)
        print(
            f"Skipping {len(good_ids)} completed, rerunning {len(examples_to_run)} "
            f"(empty/parse-fail: {empty}, missing: {len(examples_to_run) - empty})"
        )
        examples = examples_to_run

    # Create semaphore for concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    async def predict_with_semaphore(example, idx):
        async with semaphore:
            return await get_model_prediction(agent, example, idx)

    print(f"Generating predictions for {len(examples)} examples...")
    tasks = [predict_with_semaphore(example, idx) for idx, example in enumerate(examples)]

    # Open the JSONL file in append mode for incremental writes
    output_path = Path(output_file)
    jsonl_file = open(output_path, 'a')
    write_lock = asyncio.Lock()

    correct = 0
    processed = 0
    pbar = tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating")
    for task in pbar:
        result = await task

        # Write result to JSONL immediately (retryable if extracted_answer is None)
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
             dataset: str = "PLM-Team/Spatial-Visualization-Benchmark",
             models_config: str = "configs/models.yaml",
             max_concurrent: int = 32,
             max_samples: int = None,
             redo: bool = True):
    """
    Run SpatialViz evaluation.

    Args:
        model: Model name from models.yaml
        output_file: Path to output JSONL file (required)
        dataset: HuggingFace dataset identifier
        models_config: Path to models configuration file
        max_concurrent: Maximum number of concurrent API calls
        max_samples: If set, limit evaluation to first N samples
        redo: If True (default), rerun all. If False, resume: skip examples
              with a valid parsed answer and only rerun missing/null ones.
    """
    # Download images if needed
    download_images_if_needed()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results from JSONL if resuming
    existing_results = None
    if redo:
        # Fresh run: start from an empty JSONL so we don't append to stale rows
        if output_path.exists():
            output_path.unlink()
    elif output_path.exists():
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

        # Rewrite the JSONL RETAINING every row (dedupe by id, last write wins).
        # Incomplete rows (null answer, but saved response) are kept on disk so a
        # crash mid-resume never loses their saved data; the read-back merge
        # dedupes retried rows over these stale ones.
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

    # Generate predictions (results are written incrementally to JSONL)
    results, total_examples = asyncio.run(
        generate_predictions(
            agent=model_agent,
            dataset=dataset,
            output_file=output_file,
            max_concurrent=max_concurrent,
            max_samples=max_samples,
            existing_results=existing_results
        )
    )

    # Compute metrics from the authoritative merged result set
    metrics = compute_metrics(results, total_examples)

    print("\n=== SpatialViz Results ===")
    print(f"Dataset: {dataset}")
    print(f"Accuracy: {metrics['accuracy']}% ({metrics['correct']}/{metrics['total']})")
    print(f"Evaluated: {metrics['evaluated']} | Failed/null: {metrics['failed']} | Total: {metrics['total']}")

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

    # Append a metrics summary line to the JSONL (authoritative dashboard number)
    metrics_line = {
        '_type': 'metrics_summary',
        'model': model,
        'dataset': dataset,
        'metrics': metrics,
    }
    with open(output_path, 'a') as f:
        f.write(json.dumps(metrics_line) + '\n')

    print(f"\nResults saved to {output_path} (JSONL, {len(results)} result lines + metrics summary)")


if __name__ == '__main__':
    fire.Fire(run_eval)
