# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modified evaluation library that uses keys instead of prompts for matching."""

import collections
import dataclasses
import json
from typing import Dict, Optional, Union
from ifeval_da import instructions_registry


@dataclasses.dataclass
class InputExample:
  key: int
  instruction_id_list: list[str]
  prompt: str
  kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
  instruction_id_list: list[str]
  prompt: str
  response: str
  follow_all_instructions: bool
  follow_instruction_list: list[bool]


def read_prompt_list(input_jsonl_filename):
  """Read inputs from jsonl."""
  inputs = []
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      inputs.append(
          InputExample(key=example["key"],
                       instruction_id_list=example["instruction_id_list"],
                       prompt=example["prompt"],
                       kwargs=example["kwargs"]))
  return inputs


def read_key_to_responses_dict(input_jsonl_filename):
  """Creates dictionary matching key to responses (list)."""
  return_dict = {}
  with open(input_jsonl_filename, "r") as f:
    for l in f:
      example = json.loads(l)
      key = example["key"]
      # Handle both old format (responses) and new format (response)
      if "responses" in example:
        return_dict[key] = example["responses"]
      elif "response" in example:
        return_dict[key] = [example["response"]]
      else:
        return_dict[key] = [""]  # Empty response if neither exists
  return return_dict


def write_outputs(output_jsonl_filename, outputs):
  """Writes outputs to jsonl."""
  assert outputs
  with open(output_jsonl_filename, "w") as f:
    for o in outputs:
      f.write(
          json.dumps(
              {
                  "instruction_id_list": o.instruction_id_list,
                  "prompt": o.prompt,
                  "response": o.response,
                  "follow_all_instructions": o.follow_all_instructions,
                  "follow_instruction_list": o.follow_instruction_list,
              }
          )
      )
      f.write("\n")


def test_instruction_following_strict(
    inp,
    key_to_responses,
    response_index=0
):
  """Tests response to see if instructions are followed."""
  responses = key_to_responses.get(inp.key, [""])
  # Use the specified response index, or the first one if index is out of bounds
  response = responses[response_index] if response_index < len(responses) else responses[0]
  
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    if response.strip() and instruction.check_following(response):
      is_following_list.append(True)
    else:
      is_following_list.append(False)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list)


def test_instruction_following_loose(
    inp,
    key_to_responses,
    response_index=0
):
  """Tests response for an upper bound for following instructions."""
  responses = key_to_responses.get(inp.key, [""])
  response = responses[response_index] if response_index < len(responses) else responses[0]
  
  r = response.split("\n")
  response_remove_first = "\n".join(r[1:]).strip()
  response_remove_last = "\n".join(r[:-1]).strip()
  response_remove_both = "\n".join(r[1:-1]).strip()
  revised_response = response.replace("*", "")
  revised_response_remove_first = response_remove_first.replace("*", "")
  revised_response_remove_last = response_remove_last.replace("*", "")
  revised_response_remove_both = response_remove_both.replace("*", "")
  all_responses = [
      response,
      revised_response,
      response_remove_first,
      response_remove_last,
      response_remove_both,
      revised_response_remove_first,
      revised_response_remove_last,
      revised_response_remove_both,
  ]
  instruction_list = inp.instruction_id_list
  is_following_list = []

  for index, instruction_id in enumerate(instruction_list):
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)

    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
      instruction.build_description(prompt=inp.prompt)

    is_following = False
    for r in all_responses:
      if r.strip() and instruction.check_following(r):
        is_following = True
        break

    is_following_list.append(is_following)

  return OutputExample(
      instruction_id_list=inp.instruction_id_list,
      prompt=inp.prompt,
      response=response,
      follow_all_instructions=all(is_following_list),
      follow_instruction_list=is_following_list)


def test_instruction_following_consistency(
    inp,
    key_to_responses,
):
  """Tests consistency across multiple responses for the same prompt."""
  responses = key_to_responses.get(inp.key, [""])
  
  if len(responses) < 2:
    # Can't evaluate consistency with less than 2 responses
    return None
  
  # Evaluate each response
  results_strict = []
  results_loose = []
  
  for i, response in enumerate(responses):
    result_strict = test_instruction_following_strict(inp, key_to_responses, i)
    result_loose = test_instruction_following_loose(inp, key_to_responses, i)
    results_strict.append(result_strict)
    results_loose.append(result_loose)
  
  # Calculate consistency metrics
  consistency_metrics = {
      "num_responses": len(responses),
      "strict_all_pass": all(r.follow_all_instructions for r in results_strict),
      "strict_consistency": sum(r.follow_all_instructions for r in results_strict) / len(results_strict),
      "loose_all_pass": all(r.follow_all_instructions for r in results_loose),
      "loose_consistency": sum(r.follow_all_instructions for r in results_loose) / len(results_loose),
      "per_instruction_consistency_strict": [],
      "per_instruction_consistency_loose": []
  }
  
  # Per-instruction consistency
  num_instructions = len(inp.instruction_id_list)
  for inst_idx in range(num_instructions):
    strict_passes = sum(r.follow_instruction_list[inst_idx] for r in results_strict)
    loose_passes = sum(r.follow_instruction_list[inst_idx] for r in results_loose)
    consistency_metrics["per_instruction_consistency_strict"].append(
        strict_passes / len(results_strict)
    )
    consistency_metrics["per_instruction_consistency_loose"].append(
        loose_passes / len(results_loose)
    )
  
  return consistency_metrics


def print_report(outputs):
  """Prints the evaluation report."""
  prompt_total = 0
  prompt_correct = 0
  instruction_total = 0
  instruction_correct = 0

  tier0_total = collections.defaultdict(int)
  tier0_correct = collections.defaultdict(int)

  tier1_total = collections.defaultdict(int)
  tier1_correct = collections.defaultdict(int)

  for example in outputs:
    follow_instruction_list = example.follow_instruction_list
    instruction_id_list = example.instruction_id_list

    prompt_total += 1
    if all(follow_instruction_list):
      prompt_correct += 1

    instruction_total += len(instruction_id_list)
    instruction_correct += sum(follow_instruction_list)

    for instruction_id, followed_or_not in zip(instruction_id_list,
                                               follow_instruction_list):
      tier0_total[instruction_id] += 1
      if followed_or_not:
        tier0_correct[instruction_id] += 1

    for instruction_id, followed_or_not in zip(instruction_id_list,
                                               follow_instruction_list):
      tier1_class = instruction_id.split(":")[0]
      tier1_total[tier1_class] += 1
      if followed_or_not:
        tier1_correct[tier1_class] += 1

  print(f"prompt-level: {prompt_correct}/{prompt_total}")
  print(f"instruction-level: {instruction_correct}/{instruction_total}")
  print()
  print("tier-0 instructions")
  for k in sorted(tier0_total.keys()):
    print(f"{k}: {tier0_correct[k]}/{tier0_total[k]}")
  print()
  print("tier-1 instructions")
  for k in sorted(tier1_total.keys()):
    print(f"{k}: {tier1_correct[k]}/{tier1_total[k]}")