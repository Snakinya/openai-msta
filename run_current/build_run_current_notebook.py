"""Build a Kaggle notebook that runs the current attack.py against one real model.

"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "notebook.ipynb"
ATTACK_SRC = (ROOT / "attack.py").read_text(encoding="utf-8")

BOOTSTRAP = '''import glob, os, sys
from pathlib import Path
sys.argv = [sys.argv[0]]
for c in glob.glob("/kaggle/input/**/kaggle_evaluation", recursive=True):
    root = str(Path(c).parent)
    if root not in sys.path:
        sys.path.insert(0, root)
    print("Dataset root:", root)
    break
print("Setup complete")
'''

INSTALL = '''import subprocess, sys
def _pip(*args):
    return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args]).returncode
try:
    import llama_cpp
    print("llama_cpp already present:", llama_cpp.__version__)
except Exception:
    installed = False
    for cu in ("cu124", "cu123", "cu122", "cu121"):
        url = f"https://abetlen.github.io/llama-cpp-python/whl/{cu}"
        if _pip("llama-cpp-python", "--extra-index-url", url) == 0:
            try:
                import llama_cpp
                print(f"Installed llama-cpp-python ({cu}):", llama_cpp.__version__)
                installed = True
                break
            except Exception:
                continue
    if not installed:
        _pip("llama-cpp-python")
        import llama_cpp
        print("Installed llama-cpp-python cpu:", llama_cpp.__version__)
'''

WRITE_ATTACK = "%%writefile /kaggle/working/attack.py\n" + ATTACK_SRC

RUN = '''import glob, os

def _all_ggufs():
    return glob.glob("/kaggle/input/**/*.gguf", recursive=True)

def _find_gguf(*needles):
    needles = tuple(n.lower() for n in needles)
    hits = []
    for path in _all_ggufs():
        low = path.lower()
        if all(n in low for n in needles):
            hits.append(path)
    return sorted(hits, key=len)[0] if hits else ""

print("Mounted GGUF files:")
for p in _all_ggufs():
    print("  ", p)
gpt_path = _find_gguf("gpt", "oss", "20b")
gemma_path = _find_gguf("gemma", "26b")
if gpt_path:
    os.environ["GPT_OSS_MODEL_PATH"] = gpt_path
if gemma_path:
    os.environ["GEMMA_MODEL_PATH"] = gemma_path
print("GPT_OSS_MODEL_PATH:", os.environ.get("GPT_OSS_MODEL_PATH") or "NOT FOUND")
print("GEMMA_MODEL_PATH  :", os.environ.get("GEMMA_MODEL_PATH") or "NOT FOUND")

# Set this to gpt_oss or gemma before pushing, or override in the notebook UI.
os.environ["AICOMP_MODEL_NAMES"] = os.environ.get("AICOMP_MODEL_NAMES", "gpt_oss")
print("Running model(s):", os.environ["AICOMP_MODEL_NAMES"])

from kaggle_evaluation.jed_attack_134815.jed_attack_inference_server import JEDAttackInferenceServer
JEDAttackInferenceServer().run_local_gateway()
'''

def cell(src: str, cid: str, kind: str = "code") -> dict:
    c = {"cell_type": kind, "id": cid, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

def main() -> None:
    nb = {
        "cells": [
            cell("# MSTA Run Current\n\nRuns current attack.py against a real mounted model.", "intro", "markdown"),
            cell(BOOTSTRAP, "bootstrap"),
            cell(INSTALL, "install"),
            cell(WRITE_ATTACK, "write-attack"),
            cell(RUN, "run"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
