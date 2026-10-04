"""SVG rendering for plan.json. Real room-polygon rendering lands in phase 3 (geometry/room_layout.py)."""
from pathlib import Path


def render_stub(out_path: Path, message: str) -> None:
    """Minimal valid SVG so the CLI always emits plan.svg, even before geometry exists."""
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="400" height="120">
  <rect width="100%" height="100%" fill="white"/>
  <text x="10" y="60" font-family="sans-serif" font-size="14" fill="black">{message}</text>
</svg>"""
    out_path.write_text(svg)
