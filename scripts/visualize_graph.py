"""Экспорт mermaid-схем графов из app/services/agent_graph.py (блок 6.3).

    python scripts/visualize_graph.py

Сохраняет docs/agent-graph-custom.mmd и docs/agent-graph-prebuilt.mmd.
Дополнительно пробует draw_mermaid_png() (нужен pyppeteer/playwright) —
если недоступно, остаются только .mmd (открываются на mermaid.live).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.agent_graph import custom_graph, prebuilt_graph  # noqa: E402

DOCS_DIR = Path(__file__).resolve().parents[1] / "docs"


def _export(graph, name: str) -> None:
    mmd = graph.get_graph().draw_mermaid()
    (DOCS_DIR / f"agent-graph-{name}.mmd").write_text(mmd, encoding="utf-8")
    print(f"=== {name} ===\n{mmd}")
    try:
        png = graph.get_graph().draw_mermaid_png()
        (DOCS_DIR / f"agent-graph-{name}.png").write_bytes(png)
        print(f"saved docs/agent-graph-{name}.png")
    except Exception as exc:  # noqa: BLE001 — png опционален (нужен pyppeteer/playwright)
        print(f"draw_mermaid_png() недоступен для {name}: {exc}")


def main() -> None:
    _export(custom_graph, "custom")
    _export(prebuilt_graph, "prebuilt")


if __name__ == "__main__":
    main()
