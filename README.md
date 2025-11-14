# CAIS AI Leaderboard Evals

Simple evaluation scripts for AI benchmarks with minimal dependencies.

## Table of Contents

- [Setup](#setup)
- [Repository Structure](#repository-structure)
- [Running Evaluations](#running-evaluations)
  - [HLE (Humanity's Last Exam)](#hle-humanitys-last-exam)
  - [ARC-AGI-2](#arc-agi-2)
  - [SWE-Bench Verified (Bash Only)](#swe-bench-verified-bash-only)
  - [TextQuests](#textquests)
  - [ERQA (Embodied Reasoning QA)](#erqa-embodied-reasoning-qa)
  - [IntPhys2](#intphys2)
  - [MindCube](#mindcube)
  - [VCT-Refusal](#vct-refusal)
  - [MASK](#mask)
  - [Machiavelli](#machiavelli)

## Setup

### 1. Environment Variables

Create a `.env` file from the example:

```bash
cp .env.example .env
```

Then fill in your API keys:

```bash
# Required: Hugging Face token for dataset access
HF_TOKEN="your_hf_token"

# Provider API Keys (add the ones you need)
OPENAI_API_KEY="your_openai_key"
ANTHROPIC_API_KEY="your_anthropic_key"
GEMINI_API_KEY="your_gemini_key"
OPENROUTER_API_KEY="your_openrouter_key"

# For Google Cloud/VertexAI
GOOGLE_CLOUD_PROJECT=""
GOOGLE_CLOUD_LOCATION=""
```

### 2. Install Dependencies

Install base requirements:

```bash
pip install -r requirements/base.txt
```

## Repository Structure

```
leaderboard_eval/
├── configs/
│   └── models.yaml          # Model configurations for all providers
├── shared/
│   ├── __init__.py
│   └── llm_agents.py        # LLM agent interface (OpenAI SDK compatible)
├── hle/                     # Humanity's Last Exam benchmark
├── arc_agi_2/               # ARC-AGI-2 benchmark
├── textquests/              # TextQuests benchmark
├── erqa/                    # ERQA benchmark
├── intphys2/                # IntPhys2 benchmark
├── mindcube/                # MindCube benchmark
├── vct_refusal/             # VCT-Refusal benchmark
├── mask/                    # MASK benchmark
├── machiavelli_eval/        # Machiavelli benchmark
└── requirements/            # Requirements
```

### Model Configuration

The [`configs/models.yaml`](configs/models.yaml) file contains configurations for all supported models and providers. Our [`llm_agents.py`](shared/llm_agents.py) module provides a unified interface that works with the [OpenAI SDK for Python](https://github.com/openai/openai-python), should support all available providers.

**Example configurations:**

```yaml
# OpenAI
gpt-5:
  model: openai/gpt-5
  generation_config:
    reasoning_effort: high

# Anthropic (via VertexAI)
claude-sonnet-4-5:
  model: anthropic/claude-sonnet-4-5
  generation_config:
    max_tokens: 40000
    vertexai: true
    thinking:
      type: enabled
      budget_tokens: 32000

# Google Gemini
gemini-2.5-pro:
  model: gemini/gemini-2.5-pro
  generation_config:
    reasoning_effort: high

# OpenRouter
grok-4-fast:
  model: openai/x-ai/grok-4-fast
  generation_config:
    api_key_env: OPENROUTER_API_KEY
    api_base_url: https://openrouter.ai/api/v1

# Custom endpoint (e.g., vLLM)
my-custom-model:
  model: openai/my-model-name
  generation_config:
    api_base_url: http://localhost:8000/v1
    api_key_env: API_KEY_ENV
```

For more provider examples, check [`configs/models.yaml`](configs/models.yaml).

## Running Evaluations

All benchmarks follow a consistent command-line interface:

```bash
python -m <benchmark>.<script> \
  --model <model_name> \
  --output_file <path> \
  --models_config configs/models.yaml
  --max_concurrent N
```

---

### HLE (Humanity's Last Exam)

[Paper](https://arxiv.org/abs/2501.04332) | [Website](https://lastexam.ai) | [Code](https://github.com/centerforaisafety/hle) | [Dataset](https://huggingface.co/datasets/cais/hle)

Expert-level questions across 100+ disciplines authored by researchers, professors, and PhD students worldwide. Tests LLMs' knowledge and reasoning at the human frontier with confidence calibration.

**Metrics:** Accuracy, Calibration Error

**Arguments:**
- `--model`: Model to evaluate (required)
- `--output_file`: Path to output JSON file (required)
- `--judge_model`: Model for judging answers (default: `gpt-5-mini`)
- `--dataset`: HuggingFace dataset path (default: `cais/hle`)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent requests (default: `4`)
- `--text_only`: Filter out questions with images (flag)
- `--max_samples`: Limit to N samples (optional)
- `--judge_only`: Only run judge on existing predictions (flag)

**Example:**

```bash
python -m hle.hle_eval \
  --model gpt-5-mini \
  --output_file results/hle/gpt-5-mini.json \
  --judge_model gpt-5-mini \
  --dataset cais/hle \
  --max_concurrent 128
```

<details>
<summary>Citation</summary>

[[citation.txt]](https://raw.githubusercontent.com/centerforaisafety/hle/main/citation.txt)

</details>

---

### ARC-AGI-2

[Paper](https://arxiv.org/abs/1911.01547) | [Website](https://arcprize.org/) | [Code](https://github.com/fchollet/ARC-AGI)

Abstract visual puzzles evaluating fluid intelligence—the ability to reason, solve novel problems, and adapt to new situations—rather than crystallized knowledge.

**Metrics:** Accuracy (Task-level)

**Arguments:**
- `--model`: Model to evaluate (required)
- `--output_file`: Path to output JSON file (required)
- `--dataset`: HuggingFace dataset path (default: `arc-agi-community/arc-agi-2`)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent requests
- `--max_samples`: Limit to N tasks (optional)

**Example:**

```bash
python -m arc_agi_2.arc_agi_2_eval \
  --model gpt-5-mini \
  --output_file results/arc_agi_2/gpt-5-mini.json \
  --dataset arc-agi-community/arc-agi-2 \
  --max_concurrent 64
```

<details>
<summary>Citation</summary>

```bibtex
@misc{chollet2019measure,
  title={On the Measure of Intelligence}, 
  author={François Chollet},
  year={2019},
  eprint={1911.01547},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/1911.01547}
}
```

</details>

---

### SWE-Bench Verified (Bash Only)

**🚧 Coming Soon**

[Paper](https://arxiv.org/abs/2310.06770) | [Website](https://www.swebench.com) | [Code](https://github.com/SWE-agent/SWE-agent) | [Dataset](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)

Tests LLMs' ability to solve 500 real Python GitHub issues using only a minimal bash shell (no tools or special scaffold structure). Scoring high would show a model can reliably understand and resolve real-world software engineering issues.

**Metrics:** Resolved Issues (%)

<details>
<summary>Citation</summary>

```bibtex
@inproceedings{jimenez2024swebench,
  title={{SWE}-bench: Can Language Models Resolve Real-world Github Issues?},
  author={Carlos E Jimenez and John Yang and Alexander Wettig and Shunyu Yao and Kexin Pei and Ofir Press and Karthik R Narasimhan},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=VTF8yNQM66}
}
```

</details>

---

### TextQuests

[Paper](https://arxiv.org/abs/2507.23701) | [Website](https://textquests.ai/) | [Code](https://github.com/centerforaisafety/textquests) | [Dataset](https://huggingface.co/datasets/justinphan3110/textquests)

Tests LLM agents on 25 classic exploratory text-based adventure games requiring long-horizon reasoning, multi-step planning, and trial-and-error problem solving.

**Requirements:** `pip install -r requirements/textquests.txt`

**Metrics:** Game Progress (mean across all games), Mean Harm (mean across all games)

**Arguments:**
- `--model`: Model to evaluate (required)
- `--dataset`: Game dataset path (default: `justinphan3110/textquests`)
- `--output_dir`: Directory for outputs (required)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--game_name`: Comma-separated game names or None for all games (optional)
- `--with_clues`: Include in-game clues (flag)
- `--redo`: Redo games even if results exist (flag)
- `--max_concurrent`: Max concurrent requests (default: `4`)
- `--max_steps`: Max steps per game (default: `500`)

**Example:**

```bash
python -m textquests.textquests_eval \
  --model gpt-5-mini \
  --output_dir results/textquests/gpt-5-mini_no_clues \
  --dataset justinphan3110/textquests \
  --max_concurrent 4 \
  --max_steps 500
```

<details>
<summary>Citation</summary>

```bibtex
@misc{phan2025textquestsgoodllmstextbased,
  title={TextQuests: How Good are LLMs at Text-Based Video Games?}, 
  author={Long Phan and Mantas Mazeika and Andy Zou and Dan Hendrycks},
  year={2025},
  eprint={2507.23701},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2507.23701}
}
```

</details>

---

### ERQA (Embodied Reasoning QA)

[Paper](https://arxiv.org/abs/2503.20020) | [Code](https://github.com/embodiedreasoning/ERQA) | [Dataset](https://huggingface.co/datasets/justinphan3110/erqa)

Evaluates Vision-Language Models on embodied reasoning questions critical for robotics, spanning spatial reasoning, trajectory reasoning, action reasoning, state estimation, pointing, multi-view reasoning, and task reasoning.

**Metrics:** Accuracy (overall and by question type)

**Arguments:**
- `--model`: Model to evaluate (required)
- `--output_file`: Path to output JSON file (required)
- `--dataset`: HuggingFace dataset path (default: `justinphan3110/erqa`)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent requests (default: `32`)
- `--max_samples`: Limit to N samples (optional)

**Example:**

```bash
python -m erqa.erqa_eval \
  --model gpt-5-mini \
  --output_file results/erqa/gpt-5-mini.json \
  --dataset justinphan3110/erqa \
  --max_concurrent 32
```

<details>
<summary>Citation</summary>

```bibtex
@misc{geminiroboticsteam2025geminiroboticsbringingai,
  title={Gemini Robotics: Bringing AI into the Physical World}, 
  author={Gemini Robotics Team and others},
  year={2025},
  eprint={2503.20020},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2503.20020}
}
```

</details>

---

### IntPhys2

[Paper](https://arxiv.org/abs/2506.09849) | [Website](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks) | [Code](https://github.com/facebookresearch/IntPhys2/tree/main) | [Dataset](https://huggingface.co/datasets/facebook/IntPhys2)

Evaluates LLMs' understanding of intuitive physics through video clips testing core principles like permanence, solidity, and spatio-temporal continuity. Models must identify physically plausible versus impossible scenes.

**Metrics:** Accuracy (overall, plausible videos, implausible videos)

**Arguments:**
- `--model`: Model to evaluate (required)
- `--output_file`: Path to output JSON file (required)
- `--dataset`: HuggingFace dataset path (default: `justinphan3110/intphys2`)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent requests (default: `8`)
- `--max_samples`: Limit to N samples (optional)

**Example:**

```bash
python -m intphys2.intphys2_eval \
  --model gpt-5-mini \
  --output_file results/intphys2/gpt-5-mini.json \
  --dataset justinphan3110/intphys2 \
  --max_concurrent 32
```

<details>
<summary>Citation</summary>

```bibtex
@misc{bordes2025intphys2benchmarkingintuitive,
  title={IntPhys 2: Benchmarking Intuitive Physics Understanding In Complex Synthetic Environments}, 
  author={Florian Bordes and Quentin Garrido and Justine T Kao and Adina Williams and Michael Rabbat and Emmanuel Dupoux},
  year={2025},
  eprint={2506.09849},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.09849}
}
```

</details>

---

### MindCube

[Paper](https://arxiv.org/abs/2506.21458) | [Website](https://mind-cube.github.io/) | [Code](https://github.com/mll-lab-nu/MindCube) | [Dataset](https://huggingface.co/datasets/MLL-Lab/MindCube)

Tests Vision-Language Models on spatial navigation working memory. Multi-view image groups are paired with spatial reasoning questions targeting object consistency, occluded elements, perspective-taking, and 'what-if' dynamics.

**Metrics:** Accuracy (overall and by category: around, rotation, among)

**Arguments:**
- `--model`: Model to evaluate (required)
- `--output_file`: Path to output JSON file (required)
- `--dataset`: HuggingFace dataset path (default: `justinphan3110/mindcube`)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent requests (default: `32`)
- `--max_samples`: Limit to N samples (optional)

**Example:**

```bash
python -m mindcube.mindcube_eval \
  --model gpt-5-mini \
  --output_file results/mindcube/gpt-5-mini.json \
  --dataset justinphan3110/mindcube \
  --max_concurrent 32
```

<details>
<summary>Citation</summary>

```bibtex
@misc{yin2025spatialmentalmodelinglimited,
  title={Spatial Mental Modeling from Limited Views}, 
  author={Baiqiao Yin and others},
  year={2025},
  eprint={2506.21458},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2506.21458}
}
```

</details>

---

### VCT-Refusal

Value-aligned content moderation testing if models appropriately refuse harmful requests.

**Metrics:** CAIS Safety Index Compliance Rate (100 - Refusal Rate)

**Arguments:**
- `--model`: Model to evaluate (required)
- `--output_dir`: Directory for outputs (required)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent requests (default: `5`)
- `--text_only`: Only use text-only prompts (flag)

**Example:**

```bash
python -m vct_refusal.vct_refusal_eval \
  --model gpt-5-mini \
  --output_dir vct_refusal_results \
  --max_concurrent 32
```

---

### MASK

[Paper](https://arxiv.org/abs/2405.14682) | [Website](https://cais.github.io/mask/) | [Code](https://github.com/centerforaisafety/MASK) | [Dataset](https://huggingface.co/datasets/cais/mask)

Model honesty under pressure—tests if models maintain truthfulness when explicitly pressured to lie about their beliefs.

**Metrics:** Honesty Score, CAIS Safety Index Deception (100 - Honesty)

**Arguments:**
- `--model`: Model to evaluate (required)
- `--judge_model`: Judge model for evaluation (required)
- `--lie_k`: Pressure level 1-3 (default: `1`)
- `--output_dir`: Directory for outputs (required)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent requests (default: `2`)
- `--test`: Test mode with small subset (flag)

**Example:**

```bash
python -m mask.mask_eval \
  --model gpt-5-mini \
  --judge_model gpt-5-mini \
  --lie_k 1 \
  --output_dir results/mask/gpt-5-mini \
  --max_concurrent 32
```

<details>
<summary>Citation</summary>

```bibtex
@misc{mondal2024mask,
  title={MASK: Model honesty under pressure}, 
  author={Sundararajan et al.},
  year={2024},
  eprint={2405.14682},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2405.14682}
}
```

</details>

---

### Machiavelli

[Paper](https://arxiv.org/abs/2304.03279) | [Website](https://aypan17.github.io/machiavelli/) | [Code](https://github.com/aypan17/machiavelli)

Ethical reasoning in text adventure games—tests for deception, power-seeking, and utility violations across 134 text-based Choose-Your-Own-Adventure games.

**Requirements:** `pip install -r requirements/machiavelli.txt`

**Metrics:** Deception Score

The **Deception Score** measures the frequency of deceptive actions (lying, misleading, betraying trust) taken by agents during gameplay. Each action in a game is annotated for ethical violations including deception. The score is computed as:

1. Count deceptive actions across all game trajectories
2. Normalize by the mean deception count of a random agent playing the same games
3. Higher scores indicate more deceptive behavior

**Note:** Our evaluation uses refined deception annotations where ambiguous scenarios were relabeled by Grok-4, GPT-5, and Gemini-2.5-Pro.

**Arguments:**
- `--model`: Model to evaluate (required)
- `--output_dir`: Directory for outputs (required)
- `--models_config`: Path to models config (default: `configs/models.yaml`)
- `--max_concurrent`: Max concurrent games (default: `5`)
- `--games`: Comma-separated game list (default: all test games)
- `--debug`: Enable debug mode (flag)
- `--skip_play`: Skip playing, only evaluate existing trajectories (flag)
- `--max_traj_length`: Max trajectory length (default: `150`)

**Example:**

```bash
python -m machiavelli_eval.machiavelli_eval \
  --model gpt-5-mini \
  --output_dir results/machiavelli/gpt-5-mini \
  --max_concurrent 5 \
  --max_traj_length 150
```

<details>
<summary>Citation</summary>

```bibtex
@inproceedings{pan2023rewards,
  title={Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the Machiavelli Benchmark},
  author={Alexander Pan and Chan Jun Shern and Andy Zou and Nathaniel Li and Steven Basart and Thomas Woodside and Jonathan Ng and Hanlin Zhang and Scott Emmons and Dan Hendrycks},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

</details>