#!/usr/bin/env python3
"""
evaluate_openai_api.py

Evaluate a model hosted with an OpenAI-compatible API on the IFEval benchmark.
Supports both English and Danish datasets.

Example usage:
python evaluate_openai_api.py \
    --input_data data/input_data.jsonl \
    --output_file data/responses_model.jsonl \
    --api_base http://localhost:8000/v1 \
    --model_name my-model \
    --api_key YOUR_API_KEY \
    --max_concurrent 10
"""

import argparse
import json
import time
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import aiohttp
from tqdm.asyncio import tqdm
from ifeval_da import evaluation_lib


async def generate_response(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    api_base: str,
    headers: Dict[str, str],
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[str]:
    """Generate a response using the OpenAI-compatible API."""
    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                print(f"Failed after {max_retries} attempts: {e}", file=sys.stderr)
                return None


async def process_prompt(
    semaphore: asyncio.Semaphore,
    session: aiohttp.ClientSession,
    prompt_data: Dict[str, Any],
    model: str,
    api_base: str,
    headers: Dict[str, str],
    temperature: float,
    max_tokens: int,
    index: int
) -> Dict[str, Any]:
    """Process a single prompt with semaphore control."""
    async with semaphore:
        prompt_text = prompt_data.get("prompt", "")
        
        response = await generate_response(
            session=session,
            prompt=prompt_text,
            model=model,
            api_base=api_base,
            headers=headers,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            "key": prompt_data.get("key", index),
            "instruction_id_list": prompt_data.get("instruction_id_list", []),
            "prompt": prompt_text,
            "response": response if response is not None else "",
            "kwargs": prompt_data.get("kwargs", [])
        }


async def process_dataset(
    prompts: List[Dict[str, Any]],
    model: str,
    api_base: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    max_concurrent: int
) -> List[Dict[str, Any]]:
    """Process all prompts concurrently with semaphore control."""
    semaphore = asyncio.Semaphore(max_concurrent)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_prompt(
                semaphore=semaphore,
                session=session,
                prompt_data=prompt_data,
                model=model,
                api_base=api_base,
                headers=headers,
                temperature=temperature,
                max_tokens=max_tokens,
                index=i
            )
            for i, prompt_data in enumerate(prompts)
        ]
        
        # Process all tasks with progress bar
        results = []
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating responses"):
            result = await f
            results.append(result)
        
        # Sort results by key to maintain order
        results.sort(key=lambda x: x["key"])
        return results


def load_prompts(input_file: Path) -> List[Dict[str, Any]]:
    """Load prompts from a JSONL file."""
    prompts = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    return prompts


def save_results(results: List[Dict[str, Any]], output_file: Path):
    """Save results to a JSONL file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a model via OpenAI-compatible API on IFEval benchmark"
    )
    
    # Required arguments
    parser.add_argument(
        "--input_data",
        type=Path,
        required=True,
        help="Path to input JSONL file with prompts"
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to output JSONL file for responses"
    )
    
    # API configuration
    parser.add_argument(
        "--api_base",
        type=str,
        default="https://api.openai.com/v1",
        help="Base URL for the OpenAI-compatible API"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="API key (can also be set via OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use"
    )
    
    # Generation parameters
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)"
    )
    
    # Concurrency parameters
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=10,
        help="Maximum concurrent requests (default: 10)"
    )
    
    # Evaluation parameters
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation and only generate responses"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Output directory for evaluation results (optional)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    import os
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api_key or OPENAI_API_KEY env var")
    
    # Load prompts
    print(f"Loading prompts from {args.input_data}...")
    prompts = load_prompts(args.input_data)
    print(f"Loaded {len(prompts)} prompts")
    
    # Generate responses
    print(f"Generating responses using {args.model_name}...")
    print(f"Max concurrent requests: {args.max_concurrent}")
    start_time = time.time()
    
    results = await process_dataset(
        prompts=prompts,
        model=args.model_name,
        api_base=args.api_base.rstrip('/'),
        api_key=api_key,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent
    )
    
    elapsed_time = time.time() - start_time
    print(f"Generated {len(results)} responses in {elapsed_time:.2f} seconds")
    
    # Save results
    print(f"Saving results to {args.output_file}...")
    save_results(results, args.output_file)
    
    # Print summary statistics
    successful_responses = sum(1 for r in results if r["response"])
    print(f"\nGeneration Summary:")
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Successful responses: {successful_responses}")
    print(f"  Failed responses: {len(prompts) - successful_responses}")
    print(f"  Average time per prompt: {elapsed_time / len(prompts):.2f} seconds")
    print(f"  Throughput: {len(prompts) / elapsed_time:.2f} prompts/second")
    
    # Run evaluation if requested
    if not args.skip_eval:
        print("\n" + "=" * 64)
        print("Running evaluation...")
        
        # Create prompt_to_response dict for evaluation
        prompt_to_response = {r["prompt"]: r["response"] for r in results}
        
        # Run both strict and loose evaluations
        for eval_name, eval_func in [
            ("STRICT", evaluation_lib.test_instruction_following_strict),
            ("LOOSE", evaluation_lib.test_instruction_following_loose),
        ]:
            print(f"\n{eval_name} Evaluation:")
            print("-" * 40)
            
            # Evaluate each prompt
            eval_results = []
            for prompt_data in prompts:
                # Convert dict to InputExample
                input_example = evaluation_lib.InputExample(
                    key=prompt_data.get("key", 0),
                    instruction_id_list=prompt_data.get("instruction_id_list", []),
                    prompt=prompt_data.get("prompt", ""),
                    kwargs=prompt_data.get("kwargs", [])
                )
                result = eval_func(input_example, prompt_to_response)
                eval_results.append(result)
            
            # Calculate and print metrics
            follow_all = [r.follow_all_instructions for r in eval_results]
            accuracy = sum(follow_all) / len(eval_results) if eval_results else 0
            
            print(f"Overall Accuracy: {accuracy:.2%} ({sum(follow_all)}/{len(eval_results)})")
            
            # Detailed report
            evaluation_lib.print_report(eval_results)
            
            # Save evaluation results if output directory specified
            if args.output_dir:
                output_dir = Path(args.output_dir)
                output_dir.mkdir(exist_ok=True)
                eval_output_file = output_dir / f"eval_results_{eval_name.lower()}.jsonl"
                evaluation_lib.write_outputs(str(eval_output_file), eval_results)
                print(f"Evaluation results saved to: {eval_output_file}")
    
    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())