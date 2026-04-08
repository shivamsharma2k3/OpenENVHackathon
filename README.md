---
title: CodeReviewEnv
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - code-review
  - bug-triage
  - reinforcement-learning
  - agent-evaluation
license: mit
---

# CodeReviewEnv 🔍

> **OpenEnv environment** — AI-driven code review and bug triage

An RL/agent benchmark where an AI agent reviews realistic Python pull request
diffs and must **identify bugs, classify severity, and suggest fixes** — earning
shaped rewards across the trajectory.

---

## Why this environment?

Code review is something every software engineer does daily, yet it is:
- **Genuinely hard** — requires context, domain knowledge, and multi-step reasoning
- **High-value** — missed bugs and security holes have real consequences
- **Under-served** in the agent/RL benchmark space

This environment fills that gap: a clean, containerised benchmark where agents
can be trained and evaluated on real code review skills.

---

## Tasks

| ID | Difficulty | What to find | Issues |
|----|-----------|--------------|--------|
| `easy` | 🟢 Easy | Null-dereference in a user service | 1 |
| `medium` | 🟡 Medium | Off-by-one boundary bug + missing thread lock in a rate limiter | 2 |
| `hard` | 🔴 Hard | SQL injection + unbounded memory leak in a report service | 2 |

Each task uses the same API — only the diff and ground truth change.

---

## Action Space

Each step, the agent submits **one** of:

| Action | Description |
|--------|-------------|
| `submit_comment` | Flag one issue with `severity`, `description`, `fix_suggestion` |
| `approve` | Approve the PR (ends episode) |
| `request_changes` | Request changes (ends episode) |
| `pass_step` | Skip (−0.02 penalty) |

```json
// submit_comment example
{
  "submit_comment": {
    "issue_id": "bug-1",
    "severity": "critical",
    "description": "SQL injection via unsanitised filter keys",
    "line_hint": 42,
    "fix_suggestion": "Use parameterised queries with ? placeholders"
  }
}
```

---

## Observation Space

```json
{
  "diff": "...",             // unified diff text
  "pr_title": "...",
  "pr_description": "...",
  "step": 3,
  "max_steps": 12,
  "comments_so_far": [...],  // comments submitted this episode
  "last_action_feedback": "...",
  "done": false
}
```

---

## Reward Function

| Event | Reward |
|-------|--------|
| Correct issue found | +0.30 |
| Correct severity | +0.10 |
| Fix suggestion matches ground truth | +0.10 |
| False positive | −0.08 |
| Duplicate comment | −0.05 |
| Pass step | −0.02 |
| Correct final decision × recall | up to +0.20 |

Rewards are **shaped across the full trajectory** — not binary end-of-episode.

---

## API Reference

### `POST /reset?task=<easy|medium|hard>`
Start a new episode. Returns `Observation`.

### `POST /step?task=<easy|medium|hard>`
Execute one action. Body: `Action` JSON. Returns `{observation, reward, done, info}`.

### `GET /state?task=<easy|medium|hard>`
Return full internal `EpisodeState`.

### `POST /grade?task=<easy|medium|hard>`
Compute normalised score [0, 1] for the current episode.

### `GET /tasks`
List all tasks with descriptions.

---

## Baseline Scores

| Task | Score |
|------|-------|
| easy | 0.75 |
| medium | 0.55 |
| hard | 0.40 |

Run `inference.py` to reproduce.

---

## Setup

### Local

```bash
git clone https://huggingface.co/spaces/your-username/codereview-env
cd codereview-env
pip install -r requirements.txt
python server.py          # starts on http://localhost:7860
```

### Docker

```bash
docker build -t codereview-env .
docker run -p 7860:7860 codereview-env
```

### Run Inference

```bash
export HF_TOKEN=hf_your_token
export BASE_URL=http://localhost:7860
python inference.py
```

---

## Project Structure

```
codereview-env/
├── env.py            # Core environment: models + CodeReviewEnv class
├── server.py         # FastAPI server (OpenEnv HTTP API)
├── inference.py      # Baseline inference script (mandatory)
├── tasks/
│   ├── __init__.py
│   └── registry.py   # All 3 tasks with ground truth
├── tests/
│   └── test_env.py   # Unit tests
├── openenv.yaml      # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## License

MIT
