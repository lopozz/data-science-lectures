import os
import json
import argparse

from datasets import load_dataset
from lm import run_inference_batch
from prompts import (
    UNANSWERABLE_Q_GENERATION_PROMPT,
    UNANSWERABLE_A_GENERATION_PROMPT,
)


def main():
    """
    This script is designed to generate questions and answers from a given dataset using a specified model.
    The script uses either the vLLM server or OpenAI API for inference, depending on the user's choice.
    The results are saved in a JSONL file.

    References:
    - HalluLens: LLM Hallucination Benchmark (https://github.com/facebookresearch/HalluLens/tree/80307ac6bc9fd396a38b7a0de4196b931611b728)
    """

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="google/gemma-3-1b-it", help="model to use")
    parser.add_argument("--dataset", type=str, default="ReDiX/wikipediaQA-ita", help="data to use")
    parser.add_argument("--inference_method", type=str, choices=["vllm", "openai", "ollama"], default="vllm", help="vLLM server or OpenAI API")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=int, default=0.5)
    parser.add_argument("--output_path", type=str, default="outputs.jsonl", help="where to save results")
    # fmt: on

    args = parser.parse_args()

    dataset = load_dataset(args.dataset, split="train")[: args.k]
    contexts = dataset["context"]
    prompts = []
    results = []

    q_prompts = []
    for c in contexts:
        q_prompts.append(
            UNANSWERABLE_Q_GENERATION_PROMPT.format(ref_document=c.strip())
        )

    questions = run_inference_batch(q_prompts, args)

    for c, q in zip(contexts, questions):
        prompts.append(
            UNANSWERABLE_A_GENERATION_PROMPT.format(
                ref_document=c.strip(), question=q.strip()
            )
        )

    args.temperature = 1
    answers = run_inference_batch(prompts, args)

    for q, c, a in zip(questions, contexts, answers):
        rec = {}
        rec["question"] = q
        rec["context"] = c
        rec["answer"] = a
        results.append(rec)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} records to {args.output_path}")


if __name__ == "__main__":
    main()
