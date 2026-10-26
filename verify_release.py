"""Check the prospective Git upload against an explicit release inventory."""

from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
MANIFEST = "RELEASE_MANIFEST.json"
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
    re.compile(r"(?i)\b(?:password|api_key|access_token|client_secret)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
]


def main() -> None:
    manifest = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    entries = manifest["files"]
    allowed = {entry["path"] for entry in entries} | {MANIFEST}
    errors = []
    for entry in entries:
        relative = Path(entry["path"])
        path = ROOT / relative
        if relative.is_absolute() or ".." in relative.parts or path.is_symlink():
            errors.append(f"Unsafe inventory path: {entry['path']}")
            continue
        if not path.is_file():
            errors.append(f"Missing release file: {entry['path']}")
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            errors.append(f"File changed since packaging: {entry['path']}")
        content = raw.decode("utf-8-sig")
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            errors.append(f"Possible credential in: {entry['path']} (value not displayed)")
    git = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=ROOT, capture_output=True, text=True) if shutil.which("git") else None
    if git is not None and git.returncode == 0 and Path(git.stdout.strip()).resolve() == ROOT:
        candidates = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT, check=True, capture_output=True,
        ).stdout.decode("utf-8").split("\0")
        mode = "Git tracked and upload-candidate files"
    else:
        # For the fresh archive, local experiments and private inputs are excluded.
        local_only = {".git", ".venv", "venv", "__pycache__", "outputs", "private_data"}
        candidates = [path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*")
                      if path.is_file() and not local_only.intersection(path.relative_to(ROOT).parts)]
        mode = "fresh release directory (no Git repository initialized here)"
    errors.extend(f"Unexpected upload file: {name}" for name in candidates if name and name not in allowed)
    if errors:
        raise SystemExit("Release check FAILED:\n" + "\n".join(sorted(set(errors))))
    print(f"Release check passed: {len(allowed)} approved files; {mode}.")
    print("Only examples/synthetic_tea.csv is approved as a dataset. No study data or weights are included.")


if __name__ == "__main__":
    main()
