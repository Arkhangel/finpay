from functools import lru_cache
from pathlib import Path
from jinja2 import Environment, FileSystemLoader

PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=8)
def render_system_prompt(version: str = "v1", **context) -> str:
    env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))
    return env.get_template(f"system_{version}.j2").render(**context)


def load_tool_description(tool_name: str) -> str:
    path = PROMPTS_DIR / "tools" / f"{tool_name}.md"
    return path.read_text(encoding="utf-8").strip()
