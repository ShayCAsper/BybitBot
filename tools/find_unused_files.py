# tools/find_unused_files.py
import os
import sys
import ast
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Treat these as "used" even if not imported (entry points / CLIs)
ENTRYPOINTS = {
    "main.py",
    "tools/monitor.py",
    "tools/effective_config.py",
    "tools/find_unused_files.py",
}

# Directories to ignore
IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

def is_ignored_dir(p: Path) -> bool:
    parts = set(p.parts)
    return bool(IGNORE_DIRS & parts)

def module_name_for_file(pyfile: Path) -> str | None:
    """
    Map a repo file path to a Python module name, e.g.:
      core/bot_manager.py -> core.bot_manager
      strategies/__init__.py -> strategies
    Returns None for files outside repo.
    """
    try:
        rel = pyfile.relative_to(REPO)
    except ValueError:
        return None

    if any(part.startswith(".") for part in rel.parts):
        return None

    if rel.name == "__init__.py":
        # package module (folder name)
        return ".".join(rel.parts[:-1])
    else:
        return ".".join(rel.with_suffix("").parts)

def collect_imports(pyfile: Path) -> set[str]:
    """
    Parse a file and return a set of fully-qualified modules it imports.
    Handles:
      - import package.sub.module
      - from package.sub import moduleX, moduleY
    (Relative imports are skipped unless you want to resolve package context.)
    """
    out: set[str] = set()
    try:
        src = pyfile.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src, filename=str(pyfile))
    except Exception:
        return out

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    out.add(alias.name)  # full dotted path
        elif isinstance(node, ast.ImportFrom):
            # skip relative for simplicity (most of your code uses absolute)
            if node.level and node.module is None:
                continue
            base = node.module or ""
            if base:
                out.add(base)
                # also record submodules: from pkg.sub import modX -> pkg.sub.modX
                for alias in node.names:
                    if alias.name and alias.name != "*":
                        out.add(f"{base}.{alias.name}")
    return out

def discover_repo_pyfiles() -> list[Path]:
    files = []
    for root, dirs, fnames in os.walk(REPO):
        # prune ignored dirs
        droot = Path(root)
        if is_ignored_dir(droot):
            continue
        dirs[:] = [d for d in dirs if not is_ignored_dir(droot / d)]
        for fn in fnames:
            if fn.endswith(".py"):
                files.append(Path(root) / fn)
    return files

def main():
    pyfiles = discover_repo_pyfiles()

    # map file -> module name
    file_to_mod: dict[Path, str] = {}
    for f in pyfiles:
        mod = module_name_for_file(f)
        if mod:
            file_to_mod[f] = mod

    # collect imports across the whole repo
    imported: set[str] = set()
    for f in pyfiles:
        imported |= collect_imports(f)

    # Consider a file "used" if:
    # 1) Its module name is exactly imported, OR
    # 2) It is an __init__.py’s package and any submodule of that package is imported, OR
    # 3) It’s an entry point in ENTRYPOINTS
    used_files: set[Path] = set()
    for f, mod in file_to_mod.items():
        # entry points
        rel = f.relative_to(REPO).as_posix()
        if rel in ENTRYPOINTS:
            used_files.add(f)
            continue

        # exact import hit
        if mod in imported:
            used_files.add(f)
            continue

        # package __init__.py case: mod is a package if it maps from __init__.py
        if f.name == "__init__.py":
            prefix = mod + "."
            if any(x.startswith(prefix) for x in imported):
                used_files.add(f)
                continue

    # the rest are candidates
    candidates = []
    for f in pyfiles:
        if f not in used_files:
            candidates.append(f.relative_to(REPO).as_posix())

    print("\nLikely-unused python files:\n")
    for c in sorted(candidates):
        print(" ", c)

if __name__ == "__main__":
    sys.exit(main())
