"""Guards for offline equation rendering on the scientific help page."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "tabular_ml_service" / "static"
VENDOR = STATIC / "vendor" / "mathjax"


def test_mathjax_bundle_is_vendored_for_offline_use() -> None:
    """The office intranet has no CDN access, so the bundle must ship with the service."""
    bundle = VENDOR / "tex-chtml-full.js"
    assert bundle.is_file()
    # "tex-chtml-full" embeds every TeX extension, so nothing is fetched lazily.
    assert bundle.stat().st_size > 500_000
    assert (VENDOR / "LICENSE").is_file()

    fonts = sorted((VENDOR / "output" / "chtml" / "fonts" / "woff-v2").glob("*.woff"))
    assert len(fonts) >= 15


def test_vendored_assets_are_not_git_ignored() -> None:
    """A bare ``output/`` ignore rule once hid MathJax's entire font directory.

    The files existed on the developer's disk, so every filesystem-level check
    passed while a fresh clone would have shipped without web fonts.
    """
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "src/tabular_ml_service/static/vendor/mathjax"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()

    assert any(name.endswith("tex-chtml-full.js") for name in tracked)
    assert sum(1 for name in tracked if name.endswith(".woff")) >= 15


def test_wheel_would_ship_the_vendored_bundle() -> None:
    """A non-recursive ``static/*`` glob does not reach ``static/vendor/**``."""
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"static/**/*",' in pyproject


def test_help_page_typesets_through_the_local_bundle(client) -> None:
    response = client.get("/tabular_ml/help")
    assert response.status_code == 200
    body = response.data.decode("utf-8")

    assert "vendor/mathjax/tex-chtml-full.js" in body
    assert "vendor/mathjax-config.js" in body
    assert "cdn.jsdelivr.net" not in body
    assert body.count('class="eq mathjax"') == 4
    assert "\\[" in body
    # The Unicode approximations that made the page look unprofessional.
    assert "R² = 1" not in body
    assert "zⱼ" not in body


def test_help_assets_are_served(client) -> None:
    bundle = client.get("/tabular_ml/assets/vendor/mathjax/tex-chtml-full.js")
    assert bundle.status_code == 200

    font = client.get(
        "/tabular_ml/assets/vendor/mathjax/output/chtml/fonts/woff-v2/MathJax_Math-Italic.woff"
    )
    assert font.status_code == 200
