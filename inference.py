"""
inference.py — CodeReviewEnv Baseline Inference Script
=======================================================

Runs an LLM agent against all three tasks (easy / medium / hard) and emits
the mandatory [START] / [STEP] / [END] log lines to stdout.

Environment variables
---------------------
API_BASE_URL     LLM API endpoint  (default: https://router.huggingface.co/v1)
MODEL_NAME       Model identifier  (default: Qwen/Qwen2.5-72B-Instruct)
HF_TOKEN         HuggingFace / API key
BASE_URL         CodeReviewEnv server URL  (default: http://localhost:7860)

Usage
-----
    python inference.py
    HF_TOKEN=hf_xxx BASE_URL=https://your-space.hf.space python inference.py
"""

import json
import os
import sys
import textwrap
import time
from typing import List, Optional

import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY       = os.getenv("HF_TOKEN")    or os.getenv("API_KEY", "hf_placeholder")
ENV_BASE_URL  = os.getenv("BASE_URL",    "http://localhost:7860").rstrip("/")

TASKS         = ["easy", "medium", "hard"]
MAX_STEPS     = 10
TEMPERATURE   = 0.2
MAX_TOKENS    = 512
BENCHMARK     = "codereview-env"

# ---------------------------------------------------------------------------
# Logging helpers (mandatory format)
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    # Truncate action to 120 chars to keep lines readable
    action_safe = action.replace("\n", " ")[:120]
    print(
        f"[STEP] step={step} action={action_safe} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ---------------------------------------------------------------------------
# OpenAI client helper
# ---------------------------------------------------------------------------

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent("""
You are an expert software engineer performing a code review.
You will be given a PR diff and must identify all bugs and security issues.

For each issue you find, respond with a JSON object (and nothing else) in this exact format:
{
  "action_type": "submit_comment",
  "issue_id": "bug-<n>",
  "severity": "<critical|high|medium|low>",
  "description": "<clear description of the bug>",
  "line_hint": <line number or null>,
  "fix_suggestion": "<concrete fix>"
}

When you have found all issues (or are confident there are no more), respond with:
{"action_type": "request_changes"}
or if the code looks fine:
{"action_type": "approve"}

Respond with ONLY the JSON object, no markdown fences, no extra text.
""").strip()


def build_user_prompt(obs: dict) -> str:
    comments_text = ""
    if obs.get("comments_so_far"):
        lines = []
        for c in obs["comments_so_far"]:
            lines.append(f"  - [{c['severity']}] {c['issue_id']}: {c['description']}")
        comments_text = "Issues found so far:\n" + "\n".join(lines)
    else:
        comments_text = "No issues submitted yet."

    return textwrap.dedent(f"""
PR Title: {obs['pr_title']}
PR Description: {obs['pr_description']}

=== DIFF ===
{obs['diff']}
============

Step {obs['step']} of {obs['max_steps']}
{comments_text}
Last feedback: {obs.get('last_action_feedback', '')}

Identify the next issue (or submit your final decision).
""").strip()


def call_llm(obs: dict, history: list) -> dict:
    """Call the LLM and parse a JSON action from its response."""
    user_prompt = build_user_prompt(obs)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])   # keep last 3 exchanges for context
    messages.append({"role": "user", "content": user_prompt})

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        history.append({"role": "assistant", "content": raw})
        return parsed
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
        return {"action_type": "request_changes"}


def build_action_payload(parsed: dict) -> dict:
    """Convert LLM output dict to the Action model expected by /step."""
    atype = parsed.get("action_type", "request_changes")
    if atype == "submit_comment":
        return {
            "submit_comment": {
                "issue_id":       parsed.get("issue_id", "bug-unknown"),
                "severity":       parsed.get("severity", "medium"),
                "description":    parsed.get("description", ""),
                "line_hint":      parsed.get("line_hint"),
                "fix_suggestion": parsed.get("fix_suggestion", ""),
            }
        }
    elif atype == "approve":
        return {"approve": True}
    else:
        return {"request_changes": True}


# ---------------------------------------------------------------------------
# Per-task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    history: list = []

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        # reset
        r = requests.post(f"{ENV_BASE_URL}/reset", params={"task": task_id}, timeout=30)
        r.raise_for_status()
        obs = r.json()

        for step in range(1, MAX_STEPS + 1):
            if obs.get("done"):
                break

            parsed = call_llm(obs, history)
            payload = build_action_payload(parsed)
            action_str = parsed.get("action_type", "?")
            if parsed.get("action_type") == "submit_comment":
                action_str = f"submit_comment(issue_id={parsed.get('issue_id')}, severity={parsed.get('severity')})"

            step_r = requests.post(
                f"{ENV_BASE_URL}/step",
                params={"task": task_id},
                json=payload,
                timeout=30,
            )
            step_r.raise_for_status()
            result = step_r.json()

            reward = result.get("reward", 0.0)
            done   = result.get("done", False)
            info   = result.get("info", {})
            obs    = result.get("observation", obs)

            error_str = None
            if info.get("false_positives", 0) > len(rewards):  # new FP this step
                error_str = "false_positive"

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward, done=done, error=error_str)

            if done:
                break

        # grade
        grade_r = requests.post(f"{ENV_BASE_URL}/grade", params={"task": task_id}, timeout=15)
        grade_r.raise_for_status()
        grade_data = grade_r.json()
        score = grade_data.get("score", 0.0)
        success = score >= 0.5

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"[DEBUG] CodeReviewEnv inference | model={MODEL_NAME} | server={ENV_BASE_URL}", flush=True)
    print(f"[DEBUG] Tasks: {TASKS}", flush=True)

    all_scores = {}
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Starting task: {task_id}", flush=True)
        score = run_task(task_id)
        all_scores[task_id] = score
        time.sleep(1)   # brief pause between tasks

    print(f"\n{'='*60}", flush=True)
    print("[DEBUG] === FINAL SCORES ===", flush=True)
    for tid, sc in all_scores.items():
        print(f"[DEBUG]   {tid:8s}: {sc:.3f}", flush=True)
    avg = sum(all_scores.values()) / len(all_scores)
    print(f"[DEBUG]   average : {avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
