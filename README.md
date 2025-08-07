# Danish Instruction Following Evaluation (IFEval-DA)

A Danish adaptation of Google's Instruction Following Evaluation benchmark for testing language models' ability to follow specific instructions.

## Overview

This repository contains:

- **541 Danish prompts** translated and natively verified from the original IFEval dataset
- **22 instruction types** that test various aspects of instruction following
- **Evaluation scripts** for both strict and loose evaluation criteria
- **Simple CLI** for easy evaluation of any OpenAI-compatible model

Note: This is a Danish-specific adaptation. The evaluation code has been modified to properly handle Danish text (sentence splitting, word counting, special characters, etc.) and is not compatible with English evaluation. For English evaluation, please use the [original IFEval repository](https://github.com/google-research/google-research/tree/master/instruction_following_eval).

## Installation

```bash
git clone https://github.com/rlrs/ifeval-da
cd ifeval-da
uv sync
```

## Quick Start

```bash
# Evaluate a model running locally (auto-detects model)
ifeval eval

# Evaluate a specific model
ifeval eval my-model

# Evaluate OpenAI GPT-4o
ifeval eval gpt-4o --api-base https://api.openai.com/v1

# View results
ifeval results --latest
```

## CLI Usage

The `ifeval` CLI provides a simple interface for all evaluation tasks:

### Evaluate a Model

```bash
# Basic evaluation (localhost:8000)
ifeval eval

# Auto-detect model from API
ifeval eval

# Specify model name
ifeval eval my-model

# Custom API endpoint
ifeval eval --api-base https://api.openai.com/v1 --model gpt-4o

# Quick test with 10 samples
ifeval eval --sample 10

# Adjust concurrency (default: 50)
ifeval eval --concurrent 100

# Skip evaluation (only generate responses)
ifeval eval --skip-eval
```

### Analyze Existing Responses

```bash
# Evaluate an existing response file
ifeval analyze data/responses_model.jsonl
```

### Compare Models

```bash
# Compare two evaluation results
ifeval compare results/eval_results_model1_strict_*.jsonl results/eval_results_model2_strict_*.jsonl
```

### View Results

```bash
# List all evaluation results
ifeval results

# Show details of latest result
ifeval results --latest
```

## Data Format

### Input Data

**Danish dataset** (`data/danish.jsonl`):
```json
{
  "key": 1000,
  "prompt": "Skriv et resume på 300+ ord...",
  "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
  "kwargs": [{}, {"relation": "at least", "num_words": 300}]
}
```

### Response Data

Generated responses include the model output:
```json
{
  "key": 1000,
  "prompt": "Skriv et resume på 300+ ord...",
  "response": "Model's response here...",
  "instruction_id_list": [...],
  "kwargs": [...]
}
```

## Instruction Types

The benchmark tests 22 different instruction types across 9 categories:

- **Keywords** (3 types): existence, frequency, forbidden words
- **Language** (1 type): response language
- **Length Constraints** (6 types): words, sentences, paragraphs, etc.
- **Detectable Content** (2 types): postscript, placeholders
- **Detectable Format** (4 types): title, sections, bullets, JSON
- **Punctuation** (1 type): no commas
- **Start/End** (2 types): end phrase, quotation
- **Change Case** (2 types): lowercase, capital frequency
- **Combination** (1 type): multiple requirements

## Evaluation Metrics

The evaluation provides:
- **Prompt-level accuracy**: Percentage of prompts where ALL instructions were followed
- **Instruction-level accuracy**: Percentage of individual instructions followed
- **Per-category breakdown**: Performance on each instruction category
- **Strict vs Loose evaluation**: Loose evaluation is more forgiving of formatting

## Manual Verification Status

- ✅ Lines 1-229: Manually verified and corrected
- ⚠️ Lines 230-541: Machine translated, pending verification

## Example Results

```
STRICT Evaluation:
Overall Accuracy: 44.5% (241/541)

Instruction-level breakdown:
change_case:      33.3% (10/30)
combination:      30.0% (3/10)
detectable_content: 85.0% (34/40)
detectable_format: 91.6% (131/143)
keywords:         84.7% (144/170)
language:         100.0% (10/10)
length_constraints: 98.5% (195/198)
punctuation:      82.0% (41/50)
startend:         70.0% (21/30)
```

## Python API

For programmatic use:

```python
from ifeval_da import evaluation_lib_key_based as evaluation_lib

# Load data
inputs = evaluation_lib.read_prompt_list("data/danish.jsonl")
responses = evaluation_lib.read_key_to_responses_dict("data/responses.jsonl")

# Run evaluation
for input_example in inputs:
    result = evaluation_lib.test_instruction_following_strict(input_example, responses)
    print(f"Prompt {input_example.key}: {result.follow_all_instructions}")
```

## Advanced Usage

### Translating New Data

To translate additional prompts to Danish:

```bash
uv run python scripts/translate_data.py \
  input_english.jsonl \
  output_danish.jsonl \
  --api_key YOUR_GEMINI_API_KEY \
  --batch_size 50
```

### Direct Script Usage

For more control, you can use the underlying evaluation module directly:

```bash
# Evaluation with all options
uv run python -m ifeval_da.evaluation_main \
  --input_data=data/danish.jsonl \
  --input_response_data=data/responses.jsonl \
  --output_dir=results/
```

## Default Settings

- **Dataset**: Danish (`data/danish.jsonl`)
- **API Base**: `http://localhost:8000/v1`
- **Concurrency**: 50 requests
- **Temperature**: 0.0
- **Max Tokens**: 2048
- **Output Directory**: `results/`

All settings can be overridden via command-line options.

## Citation

If you use this dataset, please cite both this work and the original IFEval:

```bibtex
@misc{ifeval-danish,
  title={Danish Instruction-Following Evaluation},
  author={Rasmus Larsen},
  year={2025},
  url={https://github.com/rlrs/ifeval-da}
}

@article{zhou2023instruction,
  title={Instruction-Following Evaluation for Large Language Models},
  author={Zhou, Jeffrey and Lu, Tianjian and Mishra, Swaroop and Brahma, Siddhartha and Basu, Sujoy and Luan, Yi and Zhou, Denny and Hou, Le},
  journal={arXiv preprint arXiv:2311.07911},
  year={2023}
}
```

## License

All datasets in this repository are released under the CC BY 4.0 International
license, which can be found here:
<https://creativecommons.org/licenses/by/4.0/legalcode>.  All source files in this
repository are released under the Apache 2.0 license, the text of which can be
found in the LICENSE file.

## Acknowledgments

- Original IFEval benchmark by Google Research
- Translation powered by Gemini 1.5 Flash