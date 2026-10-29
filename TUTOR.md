# ISAAC-TUTOR

> **Note:** the ISAAC-TUTOR code (`cli.py`, `tutor/`, `app/`, `build/`, `curriculum.json`) lives outside this
> repository and is not published. This file is kept as documentation of how the tutor was set up and what it
> measured on the training machine; the paths and commands below refer to that private checkout, not to files
> in this repo.

A local bot that is an expert on **NVIDIA Isaac Lab** (the `isaac-sim/IsaacLab` repository) and teaches
you reinforcement learning in a simulated environment — grounded in the actual documentation and source
code, with a citation for every claim.

Everything runs on this machine: retrieval is local (numpy + BM25), embeddings are local
(`ollama/nomic-embed-text`), generation is local (`ollama/qwen3.5`). No API keys, no cloud.

---

## What is indexed

18,724 passages from 1,907 documents:

| Source | Content | Passages |
|---|---|---|
| `repo/docs/source/**` | Official Isaac Lab docs (concepts, how-to, tutorials, policy deployment, refs) | 3,533 |
| `repo/source/**`, `scripts/**`, `examples/**`, `tools/**` | Framework code, all 148 task `*_env_cfg.py`, training entrypoints (docstrings + comments intact) | 7,958 |
| `repo/skills/**` | NVIDIA's own in-repo agent skills (`train-rl-agents`, `debug-rl-training`, `create-environments`, …) | 467 |
| `repo/README/AGENTS/CLAUDE/CONTRIBUTING` + recent `CHANGELOG` | Repo-level guidance and release notes | 74 |
| `docs.isaacsim.omniverse.nvidia.com` (latest) | **All 475 pages** of the Isaac Sim documentation, crawled once | 6,677 |
| `build/primer/*.md` | Four tutor-written primers (RL env contract, vectorised envs, one PPO iteration, reading a task config) — labelled as tutor-authored, not NVIDIA docs | 15 |

Pinned to **Isaac Lab v3.0.0**, repo commit `fa569a5` (2026-10-26). The tutor is told which version it is
answering for, because the 3.x CLI (`uv run isaaclab train …`) differs from the 1.x/2.x scripts
(`./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py`).

## Run it

```bash
# web app (already installed as a user service, see below)
xdg-open http://127.0.0.1:8790

# terminal
./.venv/bin/python cli.py ask "what does an RL environment actually do?"
./.venv/bin/python cli.py lesson "vectorised environments"
./.venv/bin/python cli.py quiz "reward terms"
./.venv/bin/python cli.py search "domain randomization event terms"
./.venv/bin/python cli.py            # interactive REPL (/mode, /k, /sources)
```

Service control:

```bash
systemctl --user status  isaaclab-tutor
systemctl --user restart isaaclab-tutor
journalctl --user -u isaaclab-tutor -f
```

## Train something for real

The clone at `repo/` is an installed Isaac Lab 3.0.0 environment (`uv sync` done, Newton/Warp backend, no Isaac Sim
needed for non-rendered tasks):

```bash
cd ~/isaaclab-tutor/repo
ollama stop qwen3.5                       # free 6.3 GB of VRAM first
uv run isaaclab list_envs | head -40      # every registered task id
uv run isaaclab random_agent --task Isaac-Cartpole --num_envs 16   # env sanity check
uv run isaaclab train --task Isaac-Cartpole --num_envs 1024 --max_iterations 150
uv run isaaclab play  --task Isaac-Cartpole --checkpoint latest --num_envs 16
```

Verified on this laptop: Cartpole is solved in ~15 s of training (mean reward 4.94, episodes survive the full
5 s time-out); checkpoints and TensorBoard events land in `repo/logs/rsl_rl/cartpole/<timestamp>/`.

## Modes

`learn` concept tutor · `code` line-by-line walkthrough · `train` exact commands and hardware reality ·
`debug` ranked causes and what to inspect · `quiz` Socratic questions · `lesson` full structured lesson.

Retrieval is hybrid: BM25 over stemmed text fused (RRF) with cosine similarity over `nomic-embed-text`
vectors, with per-document diversity caps so one file cannot dominate the context.

## Rebuild / refresh the knowledge

```bash
./build/update.sh          # git pull the repo, re-crawl Isaac Sim docs, re-extract, re-embed
```

Individual stages:

```bash
.venv/bin/python build/crawl_isaacsim.py   # Isaac Sim docs -> data/isaacsim_pages.jsonl
.venv/bin/python build/extract_repo.py     # repo -> data/repo_docs.jsonl
.venv/bin/python build/build_index.py      # chunk + embed -> index/{corpus.jsonl,vectors.npy,meta.json}
```

## Files

```
repo/                 IsaacLab clone (pinned, shallow)
data/                 extracted document text (repo_docs.jsonl, isaacsim_pages.jsonl)
index/                corpus.jsonl + vectors.npy + meta.json
tutor/core.py         chunking, tokeniser, BM25, hybrid search, prompts, Ollama client
app/server.py         FastAPI: /api/ask (NDJSON stream), /api/search, /api/meta, /api/file
app/static/index.html single-page chat UI
cli.py                terminal tutor
curriculum.json       the 6-stage learning path used by the UI
build/                the three build stages + html->text helper
```

## Hardware reality on this laptop (measured, not guessed)

RTX 3070 Laptop (8 GB VRAM), 14 GB RAM, driver 595.99.02. Isaac Lab 3.0.0 installs **Newton (Warp)** by
default; Isaac Sim is only needed for RTX camera rendering, so training works here without it.

| Step | Result on this machine |
|---|---|
| `uv sync --python 3.12` | OK — torch 2.11.0+cu130 (CUDA available), warp 1.17.0, ~8.1 GB in `repo/.venv`, ~12 GB uv cache |
| `isaaclab train --task Isaac-Cartpole --num_envs 64 --max_iterations 8` | OK — smoke test, 20.6 s wall clock, exit 0 |
| `isaaclab train --task Isaac-Cartpole --num_envs 1024 --max_iterations 150` | **solved** — mean reward 4.94, mean episode length 300/300 (the 5 s time-out), 15 s of training, checkpoints `model_0/50/100/149.pt` |
| `isaaclab play --task Isaac-Cartpole --checkpoint latest --num_envs 16` | OK — loads `model_149.pt`, attaches the policy, runs |

VRAM during training: ~1.3 GB for Cartpole at 1024 envs — but Ollama's 9B model holds ~6.3 GB, so stop it
(`ollama stop qwen3.5`) before starting a run; it reloads on the next tutor question.

The tutor runs comfortably: index load ~2 s, retrieval ~100 ms, generation ~35 tok/s. Anything with cameras
(Isaac Sim/RTX) does not fit in 8 GB alongside the tutor model; manipulation/locomotion without cameras
should, at reduced `num_envs`.

Logs: `repo/logs/rsl_rl/cartpole/<timestamp>/` (TensorBoard events + `model_*.pt`).

## Limits worth knowing

- Answers are only as good as retrieval: if a question has no matching passage the tutor says so instead
  of inventing a citation.
- The Isaac Sim corpus is the `latest` docs snapshot (475 pages); Isaac Lab docs come from the repo, so
  they always match the cloned commit.
- The generation model is a 9B local model — it is a strong explainer, not a substitute for reading the
  cited file. Every answer links the file and line range; click a source chip in the UI to read it.
