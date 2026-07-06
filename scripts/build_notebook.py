"""Generate the Kaggle submission notebook from ``attack.py``.

Keeps a single source of truth: the notebook simply materializes ``attack.py``
into ``/kaggle/working/`` (what the hosted evaluator loads) and runs an offline
smoke test. Regenerate with ``python scripts/build_notebook.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ATTACK_SRC = (ROOT / "attack.py").read_text(encoding="utf-8")
NOTEBOOK_PATH = ROOT / "notebook.ipynb"


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


INTRO = """# AI Agent Security — Multi-Step Tool Attacks

This notebook (1) writes `attack.py` to `/kaggle/working/` and (2) starts the
competition inference server. During the hidden rerun, `server.serve()` blocks
and the competition gateway drives the attack against GPT-OSS / Gemma under the
public and private guardrails, writing `submission.csv`. During an interactive
commit, `run_local_gateway()` runs the same flow locally against the fast
deterministic agent so the commit produces a valid `submission.csv`.

`AttackAlgorithm` explores the live tool-using agent, keeps only replay-verified
chains that trip a security predicate, amplifies them across tool-argument knobs
for novelty, and sizes its output to the replay budget. See `README.md` for the
full write-up.
"""

# NOTE: %%writefile must be the very FIRST line of the cell (no preceding
# comment), otherwise Jupyter treats it as an unknown line magic and errors.
WRITE_HEADER = "%%writefile /kaggle/working/attack.py\n"

SERVE = '''import os

# On the hidden rerun the competition gateway sets this and drives the attack
# against GPT-OSS / Gemma. Locally we run a fast deterministic gateway pass so
# the interactive commit still emits a valid submission.csv.
if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    os.environ.setdefault("AICOMP_MODEL_NAMES", "deterministic")

from kaggle_evaluation.jed_attack_134815.jed_attack_inference_server import (
    JEDAttackInferenceServer,
)

server = JEDAttackInferenceServer()
if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    server.serve()
else:
    server.run_local_gateway()
'''


def main() -> None:
    write_cell_source = WRITE_HEADER + ATTACK_SRC
    notebook = {
        "cells": [
            md_cell(INTRO, "intro"),
            code_cell(write_cell_source, "write-attack"),
            code_cell(SERVE, "serve"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {NOTEBOOK_PATH} ({NOTEBOOK_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
