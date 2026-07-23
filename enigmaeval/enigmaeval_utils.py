"""
Consolidated utilities for EnigmaEval evaluation.
Contains all necessary functions, types, and constants.
"""

import os
import re
import base64
import io
from pathlib import Path
from pydantic import BaseModel
import datasets
from PIL import Image


# =================== SCHEMA ===================

class AnnotatedPuzzle(BaseModel):
    puzzle_id: str
    puzzle_source: str
    puzzle_text: str | None
    base64_images: list[str]
    answer: str
    prev_answers: str | None


# =================== CONSTANTS ===================

SOURCES = {
    "pp": "PuzzledPint",
    "lde": "Labor Day Extravaganza",
    "mit": "MIT Mystery Hunt",
    "crums": "CRUMS",
    "cc": "Cryptic Crossword",
    "potluck": "Puzzle Potluck",
    "cs50": "CS50x",
    "gm": "GMPuzzles",
}

SPLITS = {
    "normal": [
        "PuzzledPint",
        "Cryptic Crossword",
        "Puzzle Potluck",
        "CRUMS",
        "CS50x",
    ],
    "hard": [
        "MIT Mystery Hunt",
        "Labor Day Extravaganza",
        "GMPuzzles",
    ],
    "pdf": [
        "CRUMS",
        "Cryptic Crossword",
        "PuzzledPint",
        "Labor Day Extravaganza",
        "CS50x",
        "GMPuzzles",
    ],
    "html": ["MIT Mystery Hunt", "Puzzle Potluck"],
    "all": list(SOURCES.values()),
}


# =================== IMAGE UTILITIES ===================

def split_images(img_data: str) -> list[str]:
    """
    Split underscore-separated image data into a list of individual images.
    
    The ScaleAI/EnigmaEval dataset stores multiple images as:
    data:image/png;base64,xxx_data:image/png;base64,yyy
    """
    if not img_data or not isinstance(img_data, str) or not img_data.strip():
        return []
    
    # Check if it contains multiple images (underscore separator)
    if '_data:image/' in img_data:
        parts = img_data.split('_data:image/')
        images = []
        for i, part in enumerate(parts):
            if i == 0:
                # First part has the full prefix
                if part.startswith('data:image/'):
                    images.append(part)
            else:
                # Subsequent parts need the prefix added back
                images.append('data:image/' + part)
        return images
    else:
        # Single image
        if img_data.startswith('data:image/'):
            return [img_data]
        else:
            return []


def convert_gif_to_png(img_data: str) -> str:
    """
    Convert GIF image to PNG format for API compatibility.    
    """
    try:
        _, b64_part = img_data.split(',', 1)
        gif_bytes = base64.b64decode(b64_part)
        
        # Standard PIL conversion
        gif_image = Image.open(io.BytesIO(gif_bytes))
        
        # Save as PNG
        png_buffer = io.BytesIO()
        gif_image.save(png_buffer, format='PNG')
        png_b64 = base64.b64encode(png_buffer.getvalue()).decode('ascii')
        
        return f"data:image/png;base64,{png_b64}"
    except Exception:
        # If conversion fails, return original
        return img_data


# =================== DATA UTILITIES ===================

def _pil_to_base64_png(img) -> str:
    """Convert a PIL.Image (as returned by HF datasets v2) to a data: URI base64 PNG."""
    if img is None:
        return None
    buf = io.BytesIO()
    # Ensure mode is something safe for PNG
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def fetch_puzzles(args):
    """Fetch puzzles from the public EnigmaEval HuggingFace dataset.

    Schema (cais/EnigmaEval):
      task_id, category, question, answer, prev_answers,
      additional_hints, dangerous_actions, problem_url, solution_url,
      images (list of PIL.Image), images_count
    Single split: "test" (1161 rows). Gated public repo -- requires an
    HF_TOKEN with the dataset's access conditions accepted.

    `additional_hints` is informational only and is NOT injected into the
    eval prompt (matches the reference implementation in the dataset card).

    SOURCES / SPLITS at the top of this file match the dataset's `category`
    values, so --split normal / --split hard work; --split all is the default.
    """
    hf_dataset = datasets.load_dataset(
        "cais/EnigmaEval",
        split="test",
        token=os.environ.get("HF_TOKEN"),
    ).shuffle(seed=42)

    # Apply category filter only when the split maps to a known source list
    if args.split in SOURCES:
        puzzle_sources = [SOURCES[args.split]]
        hf_dataset = hf_dataset.filter(lambda example: example["category"] in puzzle_sources)
    elif args.split in SPLITS and args.split != "all":
        puzzle_sources = SPLITS[args.split]
        hf_dataset = hf_dataset.filter(lambda example: example["category"] in puzzle_sources)
    elif args.split != "all":
        raise ValueError(f"Invalid split: {args.split}")

    puzzles = []
    for row in hf_dataset:
        category = row["category"]
        puzzle_text = row["question"]
        prev_answers = row.get("prev_answers")

        # Handle meta puzzles
        if args.exclude_meta and "meta" in (category or "").lower():
            continue

        # Handle images: v2 stores them as a list of PIL.Image objects.
        imgs = []
        images_field = row.get("images") or []
        if images_field:
            if args.text_only:
                continue  # Skip puzzles with images if text_only is requested
            for pil_img in images_field:
                converted = _pil_to_base64_png(pil_img)
                if converted is not None:
                    imgs.append(converted)

        puzzles.append(
            AnnotatedPuzzle(
                puzzle_id=row["task_id"],
                puzzle_source=category,
                puzzle_text=puzzle_text,
                base64_images=imgs,
                answer=row["answer"],
                prev_answers=prev_answers,
            )
        )

    return puzzles


# =================== PROMPT UTILITIES ===================

def get_answer_type(answer: str):
    """Determine the answer type and format based on the answer string."""
    if "," in answer:
        if answer.count(",") == 1:
            answer_type = "pair of words/phrases"
            answer_format = "answer1, answer2"
        else:
            answer_type = "comma-separated list of words/phrases"
            answer_format = "answer1, answer2, ..."
    else:
        answer_type = "word or short phrase"
        answer_format = "answer"
    
    answer_format = f"<answer>{answer_format}</answer>"
    return answer_type, answer_format


class PromptMaker:
    """Creates prompts for different puzzle types."""
    
    def __init__(self, data_dir: str):
        """Initialize with prompt template directory."""
        data_dir = Path(data_dir)
        self.standard_tips = "\n" + (data_dir / "standard_tips.txt").read_text() + "\n---\n"
        self.imageless_tips = "\n" + (data_dir / "imageless_tips.txt").read_text() + "\n---\n"
        self.mit_tips = "\n" + (data_dir / "mit_tips.txt").read_text() + "\n---\n"
        self.plagiarism_prompt = "\n" + (data_dir / "plagiarism.txt").read_text() + "\n---\n"

    def get_prompt_pdf(self, answer: str, prev_answers: list[str] | None = None):
        """Generate prompt for PDF puzzles."""
        answer_type, answer_format = get_answer_type(answer)
        if prev_answers:
            preamble = f" This puzzle uses at least some of the following answers from other puzzles to arrive at its own solution: {prev_answers}."
        else:
            preamble = ""
        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}.
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Here are some tips for solving puzzles of this type:
{self.standard_tips}
"""

        user_prompt = f"""Your task is to solve the following puzzle, originally in PDF format and attached here as images.{preamble}
Write out a step-by-step solution to the puzzle. At the end of your solution, write your answer in the following format:
{answer_format}"""

        return user_prompt, system_prompt

    def get_prompt_standard_transcribed(
        self,
        puzzle_statement: str,
        answer: str,
        prev_answers: list[str] | None = None,
    ):
        """Generate prompt for standard transcribed puzzles."""
        answer_type, answer_format = get_answer_type(answer)

        has_image = "[image_1]" in puzzle_statement
        if has_image:
            preamble = (
                " The attached images are presented in the order they are referenced in the text."
            )
        else:
            preamble = ""
        if prev_answers:
            preamble = f" This puzzle uses at least some of the following answers from other puzzles to arrive at its own solution: {prev_answers}.{preamble}"

        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}. 
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Here are some tips for solving puzzles of this type:
{self.standard_tips}
"""
        user_prompt = f"""Your task is to solve the following puzzle.{preamble}

Puzzle:
{puzzle_statement}

Write out a step-by-step solution to the puzzle. At the end of your solution, write your answer in the following format:
{answer_format}
"""
        return user_prompt, system_prompt

    def get_prompt_nocheating(
        self,
        puzzle_statement: str,
        answer: str,
        prev_answers: list[str] | None = None,
    ):
        """Generate no-cheating prompt for puzzles."""
        answer_type, answer_format = get_answer_type(answer)

        has_image = "[image_1]" in puzzle_statement
        if has_image:
            preamble = (
                " The attached images are presented in the order they are referenced in the text."
            )
        else:
            preamble = ""
        if prev_answers:
            preamble = f" This puzzle uses at least some of the following answers from other puzzles to arrive at its own solution: {prev_answers}.{preamble}"

        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}. 
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Do not search the internet for the answer to the puzzle. You must not consult any web forums or puzzle-solving websites. If you have seen this puzzle before, do not use that information to solve it.
Here are some tips for solving puzzles of this type:
{self.standard_tips}
"""
        user_prompt = f"""Your task is to solve the following puzzle.{preamble}

Puzzle:
{puzzle_statement}

Write out an original step-by-step solution to the puzzle. Each step of the solution should be based on your own reasoning. At the end of your solution, write your answer in the following format:
{answer_format}
"""
        return user_prompt, system_prompt

    def get_prompt_mit_transcribed(
        self, puzzle_statement: str, answer: str, include_general_tips: bool = True
    ):
        """Generate prompt for MIT Mystery Hunt puzzles."""
        answer_type, answer_format = get_answer_type(answer)

        has_image = "[image_1]" in puzzle_statement
        if has_image:
            preamble = (
                " The attached images are presented in the order they are referenced in the text."
            )
        else:
            preamble = ""

        system_prompt = f"""You will be presented with a puzzle to solve. The puzzle may not have specific instructions, but you know that the answer to the puzzle is a {answer_type}. 
Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Here are some tips for solving puzzles of this type:
{self.mit_tips}
"""
        user_prompt = f"""Your task is to solve the following puzzle.{preamble}

Puzzle:
{puzzle_statement}

Write out a step-by-step solution to the puzzle. At the end of your solution, write your answer in the following format:
{answer_format}
"""
        return user_prompt, system_prompt

    def get_prompt_plagiarism(self, model_response: str):
        """Generate plagiarism detection prompt."""
        return self.plagiarism_prompt.format(model_response=model_response)
