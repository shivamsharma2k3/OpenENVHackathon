"""FastAPI server for CodeReviewEnv."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import Action, CodeReviewEnv, Observation

app = FastAPI(
    title="CodeReviewEnv",
    description="OpenEnv environment for AI-driven code review and bug triage",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: Dict[str, CodeReviewEnv] = {}


def _get_env(task: str) -> CodeReviewEnv:
    if task not in _envs:
        _envs[task] = CodeReviewEnv(task_id=task)
    return _envs[task]


class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class GradeResult(BaseModel):
    task: str
    score: float
    found_issues: list
    false_positives: int
    final_decision: Optional[str]
    steps_taken: int


@app.post("/reset", response_model=Observation)
def reset(task: str = Query("easy", description="Task difficulty: easy | medium | hard")):
    """Reset the environment and return the initial observation."""
    env = _get_env(task)
    try:
        obs = env.reset()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return obs


@app.post("/step", response_model=StepResult)
def step(
    action: Action,
    task: str = Query("easy", description="Task difficulty: easy | medium | hard"),
):
    """Execute one action and return observation, reward, done, info."""
    env = _get_env(task)
    if env._state is None:
        raise HTTPException(status_code=400, detail="Call /reset first.")
    try:
        obs, reward, done, info = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StepResult(observation=obs, reward=reward, done=done, info=info)


@app.get("/state")
def state(task: str = Query("easy")):
    """Return the full internal episode state."""
    env = _get_env(task)
    return env.state()


@app.get("/tasks")
def list_tasks():
    """List available task IDs with descriptions."""
    return {
        "tasks": [
            {
                "id": "easy",
                "difficulty": "easy",
                "description": "Find one obvious null-dereference bug in a user service.",
                "num_issues": 1,
            },
            {
                "id": "medium",
                "difficulty": "medium",
                "description": "Find an off-by-one boundary bug AND a missing thread-safety lock in a rate limiter.",
                "num_issues": 2,
            },
            {
                "id": "hard",
                "difficulty": "hard",
                "description": "Find a SQL injection vulnerability AND an unbounded memory leak in a report service.",
                "num_issues": 2,
            },
        ]
    }


@app.post("/grade", response_model=GradeResult)
def grade(task: str = Query("easy")):
    """Compute normalised 0-1 score for the current (or just-finished) episode."""
    env = _get_env(task)
    if env._state is None:
        raise HTTPException(status_code=400, detail="No episode to grade. Call /reset first.")
    score = env.grade()
    state_obj = env._state
    return GradeResult(
        task=task,
        score=score,
        found_issues=state_obj.found_issue_ids,
        false_positives=state_obj.false_positives,
        final_decision=state_obj.final_decision,
        steps_taken=state_obj.step,
    )


@app.get("/health")
def health():
    return {"status": "ok", "env": "CodeReviewEnv", "version": "1.0.0"}


def main() -> None:
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
