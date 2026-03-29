from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(*parts: str) -> str:
    """Build an absolute path inside the project root."""
    return str(PROJECT_ROOT.joinpath(*parts))
