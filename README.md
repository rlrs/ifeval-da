# Danish Instruction Following Evaluation (IFEval-DA)

A Danish adaptation of Google's Instruction Following Evaluation benchmark for testing language models' ability to follow specific instructions.

## Overview

This repository contains:

- **541 Danish prompts** translated and natively verified from the original IFEval dataset
- **22 instruction types** that test various aspects of instruction following
- **Evaluation scripts** for both strict and loose evaluation criteria
- **OpenAI-compatible API support** for evaluating any model with an API

## Installation

```bash
git clone https://github.com/rlrs/ifeval-da
cd ifeval-da
uv sync
```

## Usage

### Evaluating Pre-Generated Responses

```bash
uv run python -m ifeval_da.evaluation_main \
  --input_data=data/translated.jsonl \
  --input_response_data=data/your_responses.jsonl \
  --output_dir=data/
```

### Evaluating via OpenAI-Compatible API

```bash
uv run python scripts/evaluate_openai_api.py \
  --input_data data/translated.jsonl \
  --output_file data/responses_model.jsonl \
  --api_base http://localhost:8000/v1 \
  --model_name your-model \
  --max_concurrent 20
```

### Translating New Data

```bash
uv run python scripts/translate_data.py \
  input_english.jsonl \
  output_danish.jsonl \
  --api_key YOUR_GEMINI_API_KEY \
  --batch_size 50
```

## Data Format

Input data (`translated.jsonl`):
```json
{
  "key": 1000,
  "prompt": "Skriv et resume på 300+ ord...",
  "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
  "kwargs": [{}, {"relation": "at least", "num_words": 300}]
}
```

Response data:
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

- ✅ Lines 1-229: Manually verified
- ⚠️ Lines 230-541: Machine translated, pending verification

## Example Results

```
STRICT Evaluation:
prompt-level: 241/541 (44.5%)
instruction-level: 589/661 (89.1%)

tier-1 instructions:
change_case: 10/30 (33.3%)
combination: 3/10 (30.0%)
detectable_content: 34/40 (85.0%)
detectable_format: 131/143 (91.6%)
keywords: 144/170 (84.7%)
language: 10/10 (100.0%)
length_constraints: 195/198 (98.5%)
punctuation: 41/50 (82.0%)
startend: 21/30 (70.0%)
```

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
