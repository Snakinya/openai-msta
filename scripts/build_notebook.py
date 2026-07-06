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


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


INTRO = """# AI Agent Security — Multi-Step Tool Attacks

This notebook writes `attack.py` to `/kaggle/working/` for the hosted evaluator,
then runs an optional offline smoke test against the deterministic agent.

`AttackAlgorithm` explores the live tool-using agent, keeps only replay-verified
chains that trip a security predicate, amplifies them across tool-argument knobs
for novelty, and sizes its output to the replay budget. See `README.md` for the
full write-up.
"""

WRITE_HEADER = "# Materialize the submission file the evaluator loads.\n" '%%writefile /kaggle/working/attack.py\n'

SMOKE = '''# Optional offline smoke test (safe to skip; requires the packaged SDK).
try:
    import importlib.util, sys

    spec = importlib.util.spec_from_file_location("attack", "/kaggle/working/attack.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from aicomp_sdk.agents import AgentSelection
    from aicomp_sdk.evaluation.runner import evaluate_redteam
    from aicomp_sdk.evaluation.ops import AttackGuardrailSpec
    from aicomp_sdk.guardrails.base import GuardrailBase

    spec_g = AttackGuardrailSpec(id="smoke", version="0", guardrail_factory=GuardrailBase)
    execution = evaluate_redteam(
        mod.AttackAlgorithm,
        budget_s=60,
        agent_selection=AgentSelection.DETERMINISTIC,
        env_selection="sandbox",
        attack_guardrail_spec=spec_g,
    )
    a = execution.attack
    print(f"smoke: score={a.score:.2f} raw={a.score_raw:.1f} "
          f"findings={a.findings_count} unique_cells={a.unique_cells}")
except Exception as exc:  # pragma: no cover
    print("smoke test skipped:", exc)
'''


def main() -> None:
    write_cell_source = WRITE_HEADER + ATTACK_SRC
    notebook = {
        "cells": [
            md_cell(INTRO),
            code_cell(write_cell_source),
            code_cell(SMOKE),
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
