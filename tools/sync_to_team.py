#!/usr/bin/env python3
"""Copy this lab to labs/Lab-10-RL in the team repository.

The copy gets Colab links that point to the team repository and freshly built
zips. This tools/ folder is not copied. The script stops if someone else changed
the lab folder in the team repository since the last sync.

Usage:
    python tools/sync_to_team.py          # show what would change
    python tools/sync_to_team.py --push   # commit and push to the team's main
"""
import argparse
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
LAB = "labs/Lab-10-RL"
SUBJECT = "Update the RL lab from rl-hvac-lab"
FIRST_SYNC = "a762343"  # team commit whose lab matches this repository's start
PERSONAL_URL = b"paabL/rl-hvac-lab/blob/main/"
TEAM_URL = b"milarobotlearningcourse/data_science_ift6758/blob/main/labs/Lab-10-RL/"
ZIPS = {"fr": "lab_rl_prototype.zip", "en": "rl_lab_en.zip"}


def git(repo, *args, binary=False):
    out = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True).stdout
    return out if binary else out.decode().strip()


def stop(message):
    sys.exit(f"Stopped: {message}")


def check_personal_repo():
    if git(HERE, "status", "--porcelain"):
        stop("this repository has uncommitted changes.")
    git(HERE, "fetch", "-q", "origin", "main")
    if git(HERE, "rev-parse", "HEAD") != git(HERE, "rev-parse", "origin/main"):
        stop("push this repository to its origin first.")


def check_team_repo(team):
    if git(team, "branch", "--show-current") != "main":
        stop(f"{team} is not on main.")
    if [l for l in git(team, "status", "--porcelain").splitlines() if not l.startswith("??")]:
        stop(f"{team} has uncommitted changes.")
    git(team, "pull", "-q", "--ff-only", "origin", "main")
    last = git(team, "log", "-1", "--format=%H", "--", LAB)
    subject = git(team, "log", "-1", "--format=%s", last)
    if not (subject.startswith(SUBJECT) or last.startswith(FIRST_SYNC)):
        stop(f"the team changed {LAB} in {last[:7]} ({subject}). Review it before syncing.")


def build_copy(target):
    archive = git(HERE, "archive", "--format=tar", "HEAD", binary=True)
    with tarfile.open(fileobj=BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    shutil.rmtree(target / "tools", ignore_errors=True)
    for path in target.rglob("*"):
        if path.suffix in {".md", ".ipynb"}:
            path.write_bytes(path.read_bytes().replace(PERSONAL_URL, TEAM_URL))
    for lang, name in ZIPS.items():
        folder = target / lang
        (folder / name).unlink()
        members = sorted(p for p in folder.rglob("*") if p.is_file() and "_execute" not in p.name)
        with zipfile.ZipFile(folder / name, "w", zipfile.ZIP_DEFLATED) as bundle:
            for path in members:
                info = zipfile.ZipInfo(path.relative_to(folder).as_posix(), (2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                bundle.writestr(info, path.read_bytes())


def changed_files(team, copy):
    current, new = Path(team) / LAB, Path(copy)
    names = {p.relative_to(current) for p in current.rglob("*") if p.is_file() and p.name != ".DS_Store"}
    names |= {p.relative_to(new) for p in new.rglob("*") if p.is_file()}
    def differs(name):
        a, b = current / name, new / name
        if not (a.exists() and b.exists()):
            return True
        if a.suffix == ".zip":  # compare contents, since zips embed timestamps
            with zipfile.ZipFile(a) as x, zipfile.ZipFile(b) as y:
                return sorted(x.namelist()) != sorted(y.namelist()) or any(x.read(n) != y.read(n) for n in x.namelist())
        return a.read_bytes() != b.read_bytes()
    return sorted(str(n) for n in names if differs(n))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--team", default=str(HERE.parent / "data_science_ift6758"), help="local clone of the team repository")
    parser.add_argument("--push", action="store_true", help="commit and push to the team's main")
    args = parser.parse_args()
    team = Path(args.team).resolve()

    check_personal_repo()
    check_team_repo(team)
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "lab"
        copy.mkdir()
        build_copy(copy)
        changes = changed_files(team, copy)
        if not changes:
            print("The team repository is already up to date.")
            return
        print("Files that differ:", *changes, sep="\n  ")
        if not args.push:
            print("Nothing was changed. Run again with --push to send them.")
            return
        git(team, "rm", "-rq", LAB)
        shutil.rmtree(team / LAB, ignore_errors=True)
        shutil.copytree(copy, team / LAB)
        git(team, "add", "--", LAB)
        git(team, "commit", "-q", "-m", f"{SUBJECT} {git(HERE, 'rev-parse', '--short', 'HEAD')}")
        git(team, "push", "-q", "origin", "main")
        print("Pushed", git(team, "rev-parse", "--short", "HEAD"), "to the team's main.")


if __name__ == "__main__":
    main()
