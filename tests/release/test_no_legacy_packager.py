"""Release docs and CI must not point at the deprecated packager."""
from pathlib import Path
import subprocess

REPO = Path(__file__).resolve().parents[2]
BAD = "package_project.py"


def _candidate_paths():
    for pattern in ("*.md", "docs/**/*.md", ".github/workflows/*.yml"):
        for path in REPO.glob(pattern):
            if path.is_file():
                yield path


def _gitignored(paths):
    """Local gitignored notes cannot ship, so they are not release docs."""
    relative = [path.relative_to(REPO).as_posix() for path in paths]
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--stdin", "-z"],
            cwd=REPO,
            input="\0".join(relative),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return set()
    return {line for line in proc.stdout.split("\0") if line}


def _texts():
    paths = list(_candidate_paths())
    ignored = _gitignored(paths)
    for path in paths:
        if path.relative_to(REPO).as_posix() in ignored:
            continue
        yield path, path.read_text(encoding="utf-8", errors="replace")


def test_no_release_docs_recommend_the_legacy_packager():
    offenders = []
    for path, text in _texts():
        for number, line in enumerate(text.splitlines(), 1):
            if BAD in line and "DEPRECATED" not in line:
                offenders.append(str(path.relative_to(REPO)) + ":" + str(number))
    assert not offenders, "remove legacy packager references:\n" + "\n".join(offenders)
