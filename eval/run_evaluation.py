#!/usr/bin/env python3
"""
LLM-as-judge evaluation runner (G-Eval style, reason-then-score).

Usage:
    python eval/run_evaluation.py \\
        --golden eval/golden_dataset.json \\
        --judge gpt-5.2 \\
        --out eval/runs/2024-01-15.json

The script:
  1. Loads the golden dataset.
  2. For each item calls the production model (temperature=0).
  3. Judges the answer with a separate judge model using a G-Eval prompt
     that requires reasoning BEFORE scores (reason-then-score pattern).
  4. Saves per-item results + aggregates to the output JSON file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from openai import AsyncOpenAI

_SYSTEM_PROMPT = (
    "Ты — ИИ-ассистент технической поддержки платёжного процессинга FinPay. "
    "Отвечай кратко и по делу на русском языке."
)

_JUDGE_PROMPT = """\
You are an impartial LLM evaluation judge assessing a payment processing support assistant.

## Question
{question}

## Reference Answer
{expected_answer}

## Keywords that should appear in the answer
{expected_keywords}

## Actual Answer to Evaluate
{actual_answer}

## Task
Evaluate the Actual Answer on three criteria, each on a scale from 1 (very poor) to 5 (excellent):
- relevance: Does the answer address the question that was asked?
- correctness: Is the information accurate and consistent with the reference answer?
- completeness: Does the answer cover the key points from the reference answer?

## Required Output Format
IMPORTANT: You MUST reason step by step through each criterion FIRST, then provide scores.
Respond ONLY with a valid JSON object in this exact structure:
{{
  "reasoning": "<step-by-step analysis covering all three criteria>",
  "relevance": <integer 1-5>,
  "correctness": <integer 1-5>,
  "completeness": <integer 1-5>,
  "explanation": "<one sentence summary of the overall quality>"
}}
"""


async def _call_production(client: AsyncOpenAI, model: str, question: str) -> str:
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        temperature=0,
        max_tokens=512,
    )
    return (response.choices[0].message.content or "").strip()


async def _judge(
    client: AsyncOpenAI,
    model: str,
    question: str,
    expected_answer: str,
    expected_keywords: list[str],
    actual_answer: str,
) -> dict:
    prompt = _JUDGE_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        expected_keywords=", ".join(expected_keywords) if expected_keywords else "—",
        actual_answer=actual_answer,
    )
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = (response.choices[0].message.content or "{}").strip()
    return json.loads(raw)


async def run_evaluation(
    golden_path: str,
    judge_model: str,
    out_path: str,
    prod_model: str | None = None,
) -> None:
    golden_text = Path(golden_path).read_text(encoding="utf-8")
    golden = json.loads(golden_text)
    items = golden["items"]
    golden_version = golden.get("version", 1)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = prod_model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

    client = AsyncOpenAI(api_key=api_key, timeout=60)

    run_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    result_items = []
    for item in items:
        print(f"  [{item['id']}] calling production model...", file=sys.stderr)
        actual = await _call_production(client, model, item["question"])

        print(f"  [{item['id']}] judging with {judge_model}...", file=sys.stderr)
        scores_raw = await _judge(
            client,
            judge_model,
            item["question"],
            item["expected_answer"],
            item.get("expected_keywords", []),
            actual,
        )

        result_items.append({
            "id": item["id"],
            "question": item["question"],
            "answer": actual,
            "scores": {
                "relevance": int(scores_raw.get("relevance", 1)),
                "correctness": int(scores_raw.get("correctness", 1)),
                "completeness": int(scores_raw.get("completeness", 1)),
            },
            "reasoning": scores_raw.get("reasoning", ""),
            "explanation": scores_raw.get("explanation", ""),
        })

    n = len(result_items)
    rel = [it["scores"]["relevance"] for it in result_items]
    cor = [it["scores"]["correctness"] for it in result_items]
    com = [it["scores"]["completeness"] for it in result_items]

    aggregates = {
        "relevance_avg": round(sum(rel) / n, 2),
        "correctness_avg": round(sum(cor) / n, 2),
        "completeness_avg": round(sum(com) / n, 2),
        "min_correctness": min(cor),
    }

    output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "model_under_test": model,
        "judge_model": judge_model,
        "golden_version": golden_version,
        "items": result_items,
        "aggregates": aggregates,
    }

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nRun saved: {out_path}", file=sys.stderr)
    print(f"  correctness_avg : {aggregates['correctness_avg']}", file=sys.stderr)
    print(f"  relevance_avg   : {aggregates['relevance_avg']}", file=sys.stderr)
    print(f"  completeness_avg: {aggregates['completeness_avg']}", file=sys.stderr)
    print(f"  min_correctness : {aggregates['min_correctness']}", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run LLM evaluation with G-Eval judge (reason-then-score)."
    )
    p.add_argument("--golden", required=True, help="Path to golden_dataset.json")
    p.add_argument("--judge", required=True, help="Judge model name (e.g. gpt-5.2)")
    p.add_argument("--out", required=True, help="Output path for run results JSON")
    p.add_argument(
        "--model",
        default=None,
        help="Production model to evaluate (overrides OPENAI_MODEL env var)",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(run_evaluation(args.golden, args.judge, args.out, args.model))


if __name__ == "__main__":
    main()
