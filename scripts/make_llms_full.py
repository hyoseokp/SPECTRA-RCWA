"""Write llms-full.txt, the agent guide and the README in one file for language models: python scripts/make_llms_full.py"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
parts = [(ROOT / ("agents/" + name if name.startswith("llms") else name)).read_text(encoding="utf-8").strip() for name in ("llms.txt", "AGENTS.md", "README.md")]
(ROOT / "agents" / "llms-full.txt").write_text("\n\n---\n\n".join(parts) + "\n", encoding="utf-8", newline="\n")
