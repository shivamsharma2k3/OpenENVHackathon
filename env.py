"""
CodeReviewEnv — OpenEnv environment for AI-driven code review and bug triage.

An agent is given a pull request diff and must:
  1. Identify all bugs / issues present
  2. Classify each issue by severity (critical / high / medium / low)
  3. Suggest a concrete fix for each issue

The environment tracks which issues have been found, rewards partial progress,
and penalizes hallucinated (non-existent) issues.
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from tasks.registry import TASK_REGISTRY


# ---------------------------------------------------------------------------
# Typed OpenEnv models
# ---------------------------------------------------------------------------

class ReviewComment(BaseModel):
    """A single review comment emitted by the agent."""
    issue_id: str = Field(..., description="Agent-assigned identifier, e.g. 'bug-1'")
    severity: str = Field(..., description="One of: critical, high, medium, low")
    description: str = Field(..., description="Human-readable description of the issue")
    line_hint: Optional[int] = Field(None, description="Approximate line number in the diff")
    fix_suggestion: str = Field(..., description="Concrete fix or recommendation")


class Action(BaseModel):
    """
    Agent action.  Exactly one of the fields should be non-null per step.

    - submit_comment : add a review comment for one issue
    - approve        : mark the PR as safe to merge (ends episode)
    - request_changes: mark the PR as needing changes (ends episode)
    - pass_step      : do nothing this step (costs a small time penalty)
    """
    submit_comment: Optional[ReviewComment] = None
    approve: Optional[bool] = None          # True → approve PR
    request_changes: Optional[bool] = None  # True → request changes
    pass_step: Optional[bool] = None


class Observation(BaseModel):
    """What the agent sees each step."""
    diff: str = Field(..., description="The full PR diff text")
    pr_title: str
    pr_description: str
    step: int
    max_steps: int
    comments_so_far: List[ReviewComment] = Field(default_factory=list)
    last_action_feedback: str = Field("", description="Feedback on last action")
    done: bool = False


class Reward(BaseModel):
    value: float
    breakdown: Dict[str, float] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    """Full internal state (returned by state())."""
    task_id: str
    step: int
    max_steps: int
    done: bool
    pr_title: str
    pr_description: str
    diff: str
    comments: List[ReviewComment] = Field(default_factory=list)
    found_issue_ids: List[str] = Field(default_factory=list)   # ground-truth IDs confirmed found
    false_positives: int = 0
    final_decision: Optional[str] = None   # "approve" | "request_changes" | None
    cumulative_reward: float = 0.0
    started_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class CodeReviewEnv:
    """
    OpenEnv-compliant code review environment.

    Lifecycle
    ---------
    env = CodeReviewEnv(task_id="easy")
    obs  = env.reset()
    while not obs.done:
        action = agent_policy(obs)
        obs, reward, done, info = env.step(action)
    """

    MAX_STEPS = 12

    def __init__(self, task_id: str = "easy"):
        if task_id not in TASK_REGISTRY:
            raise ValueError(f"Unknown task '{task_id}'. Choose from: {list(TASK_REGISTRY)}")
        self.task_id = task_id
        self._task = TASK_REGISTRY[task_id]
        self._state: Optional[EpisodeState] = None

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Start a fresh episode. Returns initial observation."""
        self._state = EpisodeState(
            task_id=self.task_id,
            step=0,
            max_steps=self.MAX_STEPS,
            done=False,
            pr_title=self._task["pr_title"],
            pr_description=self._task["pr_description"],
            diff=self._task["diff"],
        )
        return self._make_obs("Welcome. Review the diff and submit comments for each issue you find.")

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        """
        Execute one agent action.

        Returns
        -------
        observation, reward_value, done, info_dict
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step()")
        if self._state.done:
            raise RuntimeError("Episode is done. Call reset() to start a new one.")

        s = self._state
        s.step += 1

        reward_val, breakdown, feedback = self._process_action(action)

        s.cumulative_reward += reward_val

        # Terminal conditions
        if action.approve or action.request_changes:
            s.done = True
            s.final_decision = "approve" if action.approve else "request_changes"
            # Bonus/penalty for correct final decision
            bonus, bonus_info = self._final_decision_bonus()
            reward_val += bonus
            s.cumulative_reward += bonus
            breakdown.update(bonus_info)
            feedback += f" | Final decision reward: {bonus:+.2f}"

        if s.step >= s.max_steps and not s.done:
            s.done = True
            feedback += " | Max steps reached — episode ended."

        obs = self._make_obs(feedback)
        info = {
            "breakdown": breakdown,
            "found_issues": s.found_issue_ids,
            "false_positives": s.false_positives,
            "cumulative_reward": s.cumulative_reward,
        }
        return obs, reward_val, s.done, info

    def state(self) -> dict:
        """Return full internal state as a dict."""
        if self._state is None:
            return {}
        return self._state.model_dump()

    # ------------------------------------------------------------------
    # Grader — call after episode ends to get normalised 0-1 score
    # ------------------------------------------------------------------

    def grade(self) -> float:
        """
        Compute a normalised score in [0, 1] for the completed episode.

        Score components:
          - Issue recall   : fraction of ground-truth issues found     (50 %)
          - Severity accuracy: fraction of found issues with correct severity (20 %)
          - Fix quality    : keyword-match proxy for fix suggestions    (20 %)
          - Decision bonus : correct approve/request_changes            (10 %)
        """
        if self._state is None:
            return 0.0

        s = self._state
        gt_issues = self._task["ground_truth_issues"]   # list of dicts
        n_gt = len(gt_issues)
        if n_gt == 0:
            return 1.0

        found_ids = set(s.found_issue_ids)

        # --- recall ---
        recall = len(found_ids) / n_gt

        # --- severity accuracy & fix quality ---
        severity_hits = 0
        fix_hits = 0
        for gt in gt_issues:
            if gt["id"] not in found_ids:
                continue
            # find the agent comment that matched
            for c in s.comments:
                if _comment_matches_issue(c, gt):
                    if c.severity.lower() == gt["severity"].lower():
                        severity_hits += 1
                    # keyword check for fix quality
                    fix_kws = gt.get("fix_keywords", [])
                    if fix_kws:
                        agent_fix = (c.fix_suggestion + " " + c.description).lower()
                        if any(kw.lower() in agent_fix for kw in fix_kws):
                            fix_hits += 1
                    else:
                        fix_hits += 1  # no keywords required → full credit
                    break

        sev_score  = severity_hits / n_gt
        fix_score  = fix_hits / n_gt

        # --- false positive penalty ---
        fp_penalty = min(0.3, s.false_positives * 0.05)

        # --- decision bonus ---
        expected_decision = self._task.get("expected_decision", "request_changes")
        dec_score = 0.0
        if s.final_decision == expected_decision:
            dec_score = 1.0
        elif s.final_decision is not None:
            dec_score = 0.0
        else:
            dec_score = 0.0  # no decision made

        raw = (
            0.50 * recall
            + 0.20 * sev_score
            + 0.20 * fix_score
            + 0.10 * dec_score
            - fp_penalty
        )
        return float(max(0.0, min(1.0, raw)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_obs(self, feedback: str) -> Observation:
        s = self._state
        return Observation(
            diff=s.diff,
            pr_title=s.pr_title,
            pr_description=s.pr_description,
            step=s.step,
            max_steps=s.max_steps,
            comments_so_far=list(s.comments),
            last_action_feedback=feedback,
            done=s.done,
        )

    def _process_action(self, action: Action) -> tuple[float, dict, str]:
        s = self._state
        reward = 0.0
        breakdown: dict[str, float] = {}
        feedback = ""

        if action.pass_step:
            reward = -0.02   # small cost for wasting a step
            breakdown["pass_penalty"] = reward
            feedback = "Pass noted. No progress made this step."

        elif action.submit_comment:
            c = action.submit_comment
            s.comments.append(c)
            r, bd, fb = self._score_comment(c)
            reward += r
            breakdown.update(bd)
            feedback = fb

        elif action.approve is True or action.request_changes is True:
            feedback = "Decision submitted."

        else:
            reward = -0.05
            breakdown["invalid_action"] = reward
            feedback = "Invalid action structure. Use submit_comment, approve, request_changes, or pass_step."

        breakdown["step_reward"] = reward
        return reward, breakdown, feedback

    def _score_comment(self, comment: ReviewComment) -> tuple[float, dict, str]:
        """Reward a single comment against ground truth."""
        s = self._state
        gt_issues = self._task["ground_truth_issues"]

        # Check if this comment corresponds to a real issue
        for gt in gt_issues:
            if _comment_matches_issue(comment, gt):
                if gt["id"] in s.found_issue_ids:
                    return -0.05, {"duplicate_penalty": -0.05}, f"Issue '{gt['id']}' already found — duplicate comment penalised."

                s.found_issue_ids.append(gt["id"])

                r = 0.3  # base for finding the issue
                bd: dict[str, float] = {"issue_found": 0.3}

                # severity bonus
                if comment.severity.lower() == gt["severity"].lower():
                    r += 0.1
                    bd["severity_correct"] = 0.1
                else:
                    r -= 0.05
                    bd["severity_wrong"] = -0.05

                # fix quality bonus
                fix_kws = gt.get("fix_keywords", [])
                agent_text = (comment.fix_suggestion + " " + comment.description).lower()
                if fix_kws and any(kw.lower() in agent_text for kw in fix_kws):
                    r += 0.1
                    bd["fix_quality"] = 0.1

                return r, bd, f"✓ Found real issue '{gt['id']}' (severity: {gt['severity']}). Reward: {r:+.2f}"

        # No match → false positive
        s.false_positives += 1
        return -0.08, {"false_positive": -0.08}, f"✗ No matching ground-truth issue for comment '{comment.issue_id}'. False positive penalised."

    def _final_decision_bonus(self) -> tuple[float, dict]:
        expected = self._task.get("expected_decision", "request_changes")
        s = self._state
        n_gt = len(self._task["ground_truth_issues"])
        recall = len(s.found_issue_ids) / max(n_gt, 1)

        if s.final_decision == expected:
            bonus = 0.2 * recall   # scales with how many issues were found
            return bonus, {"decision_correct_bonus": bonus}
        else:
            return -0.1, {"decision_wrong_penalty": -0.1}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _comment_matches_issue(comment: ReviewComment, gt: dict) -> bool:
    """
    Heuristic match: checks if the comment's description or issue_id contains
    any of the ground-truth keywords.
    """
    keywords: list[str] = gt.get("match_keywords", [])
    if not keywords:
        return False
    text = (comment.description + " " + comment.issue_id + " " + comment.fix_suggestion).lower()
    # Require at least 2 keyword hits for robustness (or 1 if only 1 keyword defined)
    hits = sum(1 for kw in keywords if kw.lower() in text)
    threshold = min(2, len(keywords))
    return hits >= threshold
