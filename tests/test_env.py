"""
tests/test_env.py — Unit tests for CodeReviewEnv

Run with:  python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from env import Action, CodeReviewEnv, ReviewComment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_comment(issue_id, severity, description, fix):
    return ReviewComment(
        issue_id=issue_id,
        severity=severity,
        description=description,
        fix_suggestion=fix,
    )


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_reset_returns_observation(self):
        env = CodeReviewEnv("easy")
        obs = env.reset()
        assert obs.step == 0
        assert not obs.done
        assert "diff" in obs.diff
        assert obs.max_steps == 12

    def test_state_after_reset(self):
        env = CodeReviewEnv("easy")
        env.reset()
        s = env.state()
        assert s["task_id"] == "easy"
        assert s["step"] == 0
        assert not s["done"]

    def test_step_before_reset_raises(self):
        env = CodeReviewEnv("easy")
        with pytest.raises(RuntimeError):
            env.step(Action(pass_step=True))

    def test_step_after_done_raises(self):
        env = CodeReviewEnv("easy")
        env.reset()
        env.step(Action(approve=True))
        with pytest.raises(RuntimeError):
            env.step(Action(pass_step=True))

    def test_max_steps_terminates_episode(self):
        env = CodeReviewEnv("easy")
        env.reset()
        for _ in range(12):
            if env._state.done:
                break
            obs, _, done, _ = env.step(Action(pass_step=True))
        assert env._state.done

    def test_all_task_ids(self):
        for task_id in ["easy", "medium", "hard"]:
            env = CodeReviewEnv(task_id)
            obs = env.reset()
            assert obs.diff != ""

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            CodeReviewEnv("impossible_task")


# ---------------------------------------------------------------------------
# Easy task grading
# ---------------------------------------------------------------------------

class TestEasyTask:
    def setup_method(self):
        self.env = CodeReviewEnv("easy")
        self.env.reset()

    def test_correct_comment_gives_positive_reward(self):
        action = Action(submit_comment=make_comment(
            issue_id="bug-1",
            severity="high",
            description="send_welcome_email does not check for None before accessing user.email",
            fix="Add a None guard: if user is None: return False",
        ))
        _, reward, _, info = self.env.step(action)
        assert reward > 0, f"Expected positive reward, got {reward}"
        assert "null-deref-send-welcome" in self.env._state.found_issue_ids

    def test_false_positive_penalised(self):
        action = Action(submit_comment=make_comment(
            issue_id="bug-fp",
            severity="low",
            description="completely made up issue about a totally nonexistent problem",
            fix="nothing",
        ))
        _, reward, _, info = self.env.step(action)
        assert reward < 0
        assert self.env._state.false_positives == 1

    def test_duplicate_comment_penalised(self):
        good = Action(submit_comment=make_comment(
            issue_id="bug-1",
            severity="high",
            description="None check missing on user object in send_welcome_email",
            fix="Guard with if user is None: return False",
        ))
        self.env.step(good)
        _, reward2, _, _ = self.env.step(good)
        assert reward2 < 0, "Duplicate comment should be penalised"

    def test_approve_ends_episode(self):
        _, _, done, _ = self.env.step(Action(approve=True))
        assert done
        assert self.env._state.final_decision == "approve"

    def test_request_changes_ends_episode(self):
        _, _, done, _ = self.env.step(Action(request_changes=True))
        assert done
        assert self.env._state.final_decision == "request_changes"

    def test_full_easy_episode_grade(self):
        """Perfect agent should score >= 0.7 on easy task."""
        self.env.step(Action(submit_comment=make_comment(
            issue_id="bug-1",
            severity="high",
            description="send_welcome_email does not null-check user before accessing user.email",
            fix="Add: if user is None: return False before accessing user attributes",
        )))
        self.env.step(Action(request_changes=True))
        score = self.env.grade()
        assert score >= 0.7, f"Expected score >= 0.7, got {score}"

    def test_no_action_episode_scores_zero(self):
        """Agent that does nothing and approves should score poorly."""
        # Reset fresh
        self.env.reset()
        self.env.step(Action(approve=True))
        score = self.env.grade()
        assert score < 0.3


# ---------------------------------------------------------------------------
# Medium task
# ---------------------------------------------------------------------------

class TestMediumTask:
    def setup_method(self):
        self.env = CodeReviewEnv("medium")
        self.env.reset()

    def test_find_both_issues(self):
        self.env.step(Action(submit_comment=make_comment(
            issue_id="bug-1",
            severity="medium",
            description="Off-by-one boundary: uses <= cutoff instead of < cutoff when evicting timestamps",
            fix="Change <= to strictly less than to correctly preserve boundary timestamps",
        )))
        self.env.step(Action(submit_comment=make_comment(
            issue_id="bug-2",
            severity="high",
            description="Missing threading lock on _buckets dict — race condition with concurrent callers",
            fix="Add threading.Lock() and wrap setdefault/append in a lock context",
        )))
        self.env.step(Action(request_changes=True))
        score = self.env.grade()
        assert score >= 0.65, f"Expected >= 0.65 for finding both issues, got {score}"

    def test_partial_credit_one_issue(self):
        self.env.step(Action(submit_comment=make_comment(
            issue_id="bug-1",
            severity="high",
            description="thread-safety missing lock race condition _buckets concurrent",
            fix="Add threading.Lock() synchronized",
        )))
        self.env.step(Action(request_changes=True))
        score = self.env.grade()
        # Should get partial credit for finding one of two issues
        assert 0.1 < score < 0.8


# ---------------------------------------------------------------------------
# Hard task
# ---------------------------------------------------------------------------

class TestHardTask:
    def setup_method(self):
        self.env = CodeReviewEnv("hard")
        self.env.reset()

    def test_sql_injection_found(self):
        action = Action(submit_comment=make_comment(
            issue_id="sec-1",
            severity="critical",
            description="SQL injection via f-string interpolation of filter keys and values in _run_report_query",
            fix="Use parameterised queries with ? placeholders instead of f-string interpolation",
        ))
        _, reward, _, _ = self.env.step(action)
        assert reward > 0
        assert "sql-injection-filter-keys" in self.env._state.found_issue_ids

    def test_memory_leak_found(self):
        action = Action(submit_comment=make_comment(
            issue_id="perf-1",
            severity="high",
            description="Unbounded cache memory leak: _REPORT_CACHE grows forever with no TTL or max size eviction",
            fix="Use functools.lru_cache with maxsize or add a TTL-based eviction policy",
        ))
        _, reward, _, _ = self.env.step(action)
        assert reward > 0
        assert "unbounded-cache-memory-leak" in self.env._state.found_issue_ids

    def test_grade_reward_range(self):
        """Grade must always be in [0, 1]."""
        for _ in range(5):
            self.env.step(Action(submit_comment=make_comment(
                "fp", "low", "fake issue xyz abc", "nothing"
            )))
        self.env.step(Action(request_changes=True))
        score = self.env.grade()
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_pass_step_small_penalty(self):
        env = CodeReviewEnv("easy")
        env.reset()
        _, reward, _, _ = env.step(Action(pass_step=True))
        assert reward == pytest.approx(-0.02)

    def test_severity_bonus(self):
        env = CodeReviewEnv("easy")
        env.reset()
        # Correct severity → higher reward
        _, r_correct, _, _ = env.step(Action(submit_comment=make_comment(
            "bug-1", "high",
            "None check missing on user in send_welcome_email before email access",
            "if user is None: return False",
        )))
        env.reset()
        # Wrong severity → lower reward
        _, r_wrong, _, _ = env.step(Action(submit_comment=make_comment(
            "bug-1", "low",
            "None check missing on user in send_welcome_email before email access",
            "if user is None: return False",
        )))
        assert r_correct > r_wrong


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
