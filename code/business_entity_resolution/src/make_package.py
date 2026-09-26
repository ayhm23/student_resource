"""Builds the final competition submission zip described in the root
``README.md``'s "Final Submission Package" section.

Packages ``output/matching_results.tsv`` + ``output/candidate_pairs.tsv``,
a clean copy of ``code/business_entity_resolution/`` (src/ + README.md +
requirements.txt), and the filled-in ``Documentation_template.md`` into a
single ``<TEAM_NAME>_submission.zip`` at the repo root. Kept as a single
script (not folded into any Step 1/2/3 pipeline stage) because packaging is
a release-time action, not a pipeline stage — it should be safe to re-run at
any point without touching model/data state.
"""

import zipfile
from pathlib import Path

from config import ROOT_DIR, MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH

# TODO: set this to the team's real competition team name before packaging
# the final submission -- left as a placeholder since it isn't known to
# this script.
TEAM_NAME = "team"

CODE_DIR = ROOT_DIR / "code" / "business_entity_resolution"
SRC_DIR = CODE_DIR / "src"
DOC_TEMPLATE_PATH = ROOT_DIR / "Documentation_template.md"

# Directory names that never belong in a submission zip even if someone
# drops a stray file under src/ -- defensive, since src/ is walked
# recursively rather than listed file-by-file.
EXCLUDED_DIR_NAMES = {"__pycache__", "data", "dataset", ".venv", ".git"}

# Suffixes for large regenerable intermediates or environment files that
# must never ship in the zip (the rules the task spec and this repo's own
# .gitignore both apply to data/*.parquet and DuckDB scratch files).
EXCLUDED_SUFFIXES = {".parquet", ".duckdb", ".duckdb.wal", ".pyc"}

# Filenames that look like secrets/credentials -- excluded on sight rather
# than trusted to never appear under src/.
EXCLUDED_NAMES = {".env", "credentials.json", "credentials.tsv"}


def is_excluded(path: Path) -> bool:
    """Return True if `path` must never be written into the submission zip.

    Checks every path component against the excluded-directory-name list
    (so a nested ``src/foo/__pycache__/x.pyc`` is caught, not just a
    top-level one), then the file's own name/suffix.
    """
    if any(part in EXCLUDED_DIR_NAMES for part in path.parts):
        return True
    if path.name in EXCLUDED_NAMES:
        return True
    if path.name.startswith(".env"):
        return True
    if any(path.name.endswith(suf) for suf in EXCLUDED_SUFFIXES):
        return True
    return False


def iter_src_files():
    """Yield every file under src/ that belongs in the submission zip.

    Recurses the real directory tree rather than hardcoding a file list so
    a new script added to src/ is picked up automatically; is_excluded()
    is the only gate keeping __pycache__/secrets/parquet out.
    """
    for path in sorted(SRC_DIR.rglob("*")):
        if path.is_file() and not is_excluded(path):
            yield path


def required_output_files():
    """Return the two output TSVs the submission zip must contain.

    Raises FileNotFoundError with an actionable message if either is
    missing, since a package built without them would fail the
    competition's own validator and should never be silently produced.
    """
    missing = [p for p in (MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required output file(s): "
            + ", ".join(str(p) for p in missing)
            + " -- run the Step 2 (or later) pipeline first, see "
            "code/business_entity_resolution/README.md."
        )
    return [MATCHING_RESULTS_PATH, CANDIDATE_PAIRS_PATH]


def build_package(zip_path: Path) -> Path:
    """Write the submission zip at `zip_path` and return that same path.

    Uses ZIP_DEFLATED throughout: candidate_pairs.tsv in particular is
    large (hundreds of MB, comma/tab text), and text compresses well
    enough that deflating it is worth the CPU cost versus ZIP_STORED.
    """
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src_path in required_output_files():
            zf.write(src_path, arcname=f"output/{src_path.name}")

        for src_path in iter_src_files():
            rel = src_path.relative_to(SRC_DIR)
            zf.write(src_path, arcname=f"code/business_entity_resolution/src/{rel.as_posix()}")

        zf.write(CODE_DIR / "README.md", arcname="code/business_entity_resolution/README.md")
        zf.write(CODE_DIR / "run_pipeline.py", arcname="code/business_entity_resolution/run_pipeline.py")
        zf.write(
            CODE_DIR / "requirements.txt",
            arcname="code/business_entity_resolution/requirements.txt",
        )
        zf.write(DOC_TEMPLATE_PATH, arcname="Documentation_template.md")
    return zip_path


def print_tree_and_size(zip_path: Path):
    """Print every entry in the built zip (sorted) and the zip's total size.

    Reads the entries back out of the zip itself rather than off disk, so
    what's printed is guaranteed to reflect what was actually written.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in sorted(zf.namelist()):
            print(f"  {name}")
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n{zip_path.name}: {size_mb:.1f} MB")


def main():
    """Build `<TEAM_NAME>_submission.zip` at the repo root and report on it."""
    if TEAM_NAME == "team":
        print("WARNING: TEAM_NAME is still the placeholder value -- edit "
              "the constant at the top of this file before the real submission.")

    zip_path = ROOT_DIR / f"{TEAM_NAME}_submission.zip"
    build_package(zip_path)
    print(f"Wrote {zip_path}\n")
    print_tree_and_size(zip_path)


if __name__ == "__main__":
    main()
