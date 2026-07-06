"""Local pipeline check: run AttackAlgorithm against the deterministic agent.

Against the public OptimalGuardrail the deterministic agent is unscoreable by
construction (every sensitive action is keyword- or taint-blocked), so we also
run with a permissive guardrail to prove the search/verify/score pipeline works
end to end. The hosted competition targets real LLMs where these chains apply.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aicomp_sdk.agents import AgentSelection  # noqa: E402
from aicomp_sdk.evaluation.ops import AttackGuardrailSpec  # noqa: E402
from aicomp_sdk.evaluation.runner import evaluate_redteam  # noqa: E402
from aicomp_sdk.guardrails.base import GuardrailBase  # noqa: E402
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail  # noqa: E402

from attack import AttackAlgorithm  # noqa: E402


def _run(guardrail_factory, label: str, budget_s: float = 60.0):
    spec = AttackGuardrailSpec(id=f"test_{label}", version="0", guardrail_factory=guardrail_factory)
    execution = evaluate_redteam(
        AttackAlgorithm,
        budget_s=budget_s,
        agent_selection=AgentSelection.DETERMINISTIC,
        env_selection="sandbox",
        attack_guardrail_spec=spec,
    )
    a = execution.attack
    print(
        f"[{label}] score={a.score:.2f} raw={a.score_raw:.1f} "
        f"findings={a.findings_count} unique_cells={a.unique_cells} "
        f"time={a.time_taken:.2f}s"
    )
    if a.findings:
        preds = {}
        for f in a.findings:
            for p in f["predicates"]:
                preds[p["predicate"]] = preds.get(p["predicate"], 0) + 1
        print(f"    predicate counts: {preds}")
        print(f"    sample chain: {a.findings[0]['user_messages']}")
    return a


if __name__ == "__main__":
    print("== deterministic agent + OptimalGuardrail (expected 0; unbeatable) ==")
    _run(OptimalGuardrail, "optimal")
    print("\n== deterministic agent + permissive guardrail (pipeline validation) ==")
    _run(GuardrailBase, "permissive")
