"""Report rendering: Quarto-first with Python-markdown fallback."""

from __future__ import annotations

import base64
import logging
import re
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

_CSS = """\
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 900px;
    margin: 2em auto;
    padding: 0 1em;
    line-height: 1.6;
    color: #333;
}
h1, h2, h3 { color: #2c3e50; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
}
th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
th { background-color: #f5f5f5; font-weight: 600; }
tr:nth-child(even) { background-color: #fafafa; }
img { max-width: 100%; height: auto; }
code { background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }
pre { background: #f4f4f4; padding: 1em; overflow-x: auto; border-radius: 4px; }
"""


def _embed_images(html: str, base_dir: Path) -> str:
    """Replace local image src with base64 data URIs."""

    def _replace(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            logger.warning("Image not found: %s", img_path)
            return match.group(0)
        suffix = img_path.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "gif": "image/gif", "svg": "image/svg+xml"}.get(suffix, "image/png")
        data = base64.b64encode(img_path.read_bytes()).decode()
        return f'src="data:{mime};base64,{data}"'

    return re.sub(r'src="([^"]+)"', _replace, html)


def _render_with_quarto(input_path: Path, output_format: str, output_dir: Path | None) -> Path:
    """Render using quarto (works for .md and .qmd, pdf and html)."""
    cmd = ["quarto", "render", str(input_path), "--to", output_format]
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(input_path.parent),
        timeout=300,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"Quarto failed (exit {result.returncode}):\n{stderr}")

    if result.stdout:
        for line in result.stdout.strip().split("\n"):
            logger.info("[quarto] %s", line)

    # Determine output path
    out_dir = output_dir or input_path.parent
    out_name = input_path.stem + f".{output_format}"
    out_path = out_dir / out_name
    if out_path.exists():
        return out_path

    # Quarto may place output in input dir even if output_dir specified
    alt = input_path.parent / out_name
    if alt.exists():
        return alt

    raise FileNotFoundError(f"Expected output not found: {out_path}")


def _render_md_to_html(input_path: Path, output_dir: Path | None) -> Path:
    """Fallback: render .md to HTML using python-markdown with embedded images."""
    try:
        import markdown
    except ImportError:
        raise RuntimeError(
            "python-markdown is required for HTML rendering without Quarto.\n"
            "Install with: uv pip install markdown\n"
            "Or install Quarto: https://quarto.org/docs/get-started/"
        )

    md_text = input_path.read_text(encoding="utf-8")
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "toc"])

    # Embed images as base64
    body = _embed_images(body, input_path.parent)

    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{input_path.stem}</title>
<style>{_CSS}</style>
</head>
<body>
{body}
</body>
</html>
"""
    out_dir = output_dir or input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{input_path.stem}.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


def render_report(
    input_path: Path,
    output_format: str = "pdf",
    output_dir: Path | None = None,
) -> Path:
    """Render a report file (.md or .qmd) to the specified format.

    Parameters
    ----------
    input_path : Path
        Path to the input .md or .qmd file.
    output_format : str
        Output format: "pdf" or "html".
    output_dir : Path | None
        Output directory. Defaults to input file's parent.

    Returns
    -------
    Path
        Path to the rendered output file.
    """
    input_path = Path(input_path).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()
    if suffix not in (".md", ".qmd"):
        raise ValueError(f"Unsupported input format '{suffix}'. Use .md or .qmd")

    has_quarto = shutil.which("quarto") is not None

    if has_quarto:
        return _render_with_quarto(input_path, output_format, output_dir)

    # Fallback without Quarto
    if suffix == ".qmd":
        raise RuntimeError(
            "Quarto is required to render .qmd files.\n"
            "Install from: https://quarto.org/docs/get-started/"
        )
    if output_format == "pdf":
        raise RuntimeError(
            "Quarto is required for PDF rendering.\n"
            "Install from: https://quarto.org/docs/get-started/\n"
            "Or use --format html for a Python-only fallback."
        )

    # .md + html without quarto
    return _render_md_to_html(input_path, output_dir)
