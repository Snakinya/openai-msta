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

This notebook (1) puts the competition dataset (`aicomp_sdk` / `kaggle_evaluation`)
on the import path, (2) writes `attack.py` to `/kaggle/working/`, and (3) starts
the competition inference server.

During Kaggle's hidden rerun, `serve()` blocks and the gateway drives the attack
against GPT-OSS / Gemma under the public and private guardrails, writing the real
`submission.csv`. On the visible commit we instead run the same flow locally
against the fast deterministic agent (`run_local_gateway()`), which emits a
`submission.csv` — required so the notebook can be submitted via the Kaggle CLI /
API (the UI Submit button reruns the notebook instead, so both paths score
identically on the hidden set).

`AttackAlgorithm` explores the live tool-using agent, keeps only replay-verified
chains that trip a security predicate, amplifies them across tool-argument knobs
for novelty, and sizes its output to the replay budget. See `README.md` for the
full write-up.
"""

BOOTSTRAP = '''import glob
import sys
from pathlib import Path

# Avoid argparse conflicts in the Kaggle notebook runtime.
sys.argv = [sys.argv[0]]

# The competition dataset ships aicomp_sdk/ and kaggle_evaluation/ at its root.
for candidate in glob.glob("/kaggle/input/**/kaggle_evaluation", recursive=True):
    dataset_root = str(Path(candidate).parent)
    if dataset_root not in sys.path:
        sys.path.insert(0, dataset_root)
    print(f"Dataset root: {dataset_root}")
    break

print("Setup complete")
'''

# NOTE: %%writefile must be the very FIRST line of the cell (no preceding
# comment), otherwise Jupyter treats it as an unknown line magic and errors.
WRITE_HEADER = "%%writefile /kaggle/working/attack.py\n"

SERVE = '''import os

_IS_RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))

# IMPORTANT: the gateway reads AICOMP_MODEL_NAMES at import time. On the visible
# commit we force the fast deterministic agent BEFORE importing so we don't try
# to load the 20B GPT-OSS / Gemma models locally (which fails / OOMs). On the
# hidden rerun we leave the default so the real models are evaluated.
if not _IS_RERUN:
    os.environ.setdefault("AICOMP_MODEL_NAMES", "deterministic")

from kaggle_evaluation.jed_attack_134815.jed_attack_inference_server import (
    JEDAttackInferenceServer,
)

# Hidden rerun: serve() blocks and the competition gateway drives the attack
# against GPT-OSS / Gemma under both guardrails, writing the real submission.csv.
# Visible commit: run the same flow locally against the deterministic agent so
# the commit emits a submission.csv -- required for CLI/API `competitions submit`
# ("Did not find provided Notebook Output File" otherwise). Both paths score
# identically on the hidden rerun.
if _IS_RERUN:
    JEDAttackInferenceServer().serve()
else:
    JEDAttackInferenceServer().run_local_gateway()
'''


def main() -> None:
    write_cell_source = WRITE_HEADER + ATTACK_SRC
    notebook = {
        "cells": [
            md_cell(INTRO, "intro"),
            code_cell(BOOTSTRAP, "bootstrap"),
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
