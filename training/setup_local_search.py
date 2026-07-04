"""One-time setup: assemble a locally runnable `cg` package with search_begin/
search_step/search_end support, by pairing the real native engine binary
already bundled inside the installed `kaggle_environments` package with the
fuller `cg/api.py` (+ sim.py/utils.py/game.py) from the `kiyotah/cg-lib`
Kaggle dataset.

Why this works: the locally installed kaggle_environments' cabt env ships the
real per-platform native binary (cg.dll/libcg.so/libcg.dylib) but a stripped
sim.py that only declares ctypes bindings for battle play, not search. The
cg-lib dataset ships the fuller sim.py/api.py (search_begin/search_step/
search_end/all_card_data/all_attack bindings) but only a Linux .so. Combining
the dataset's Python source with the LOCAL platform's native binary gives a
fully working search-capable `cg` package with no Kaggle round-trip needed for
MCTS development/testing. Confirmed working 2026-07-04 (see docs/engine-api.md).

This does NOT change what ships to the ladder — `main.py` still calls
`cg.api.search_begin` the normal way, which the live Kaggle evaluation
environment provides directly (Stage 5 is "Kaggle-gated" for *testing*, not
for deployment). This script only unlocks fast local dev/test iteration.

Not committed to git: cg.dll/libcg.* are third-party native binaries and
api.py/sim.py/utils.py/game.py are the competition organizer's redistributed
source — regenerate with this script instead (`local_cg/` is gitignored).

Run once from repo root: python training/setup_local_search.py
"""
import glob
import os
import platform
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "training", "local_cg")
OUT_CG = os.path.join(OUT_DIR, "cg")

NATIVE_LIB_BY_PLATFORM = {
    "Windows": "cg.dll",
    "Darwin": "libcg.dylib",
}


def find_kaggle_environments_native_lib():
    import kaggle_environments
    cabt_cg_dir = os.path.join(os.path.dirname(kaggle_environments.__file__), "envs", "cabt", "cg")
    system = platform.system()
    if system in NATIVE_LIB_BY_PLATFORM:
        name = NATIVE_LIB_BY_PLATFORM[system]
    elif platform.machine() in ("arm64", "aarch64"):
        name = "libcg-arm64.so"
    else:
        name = "libcg.so"
    path = os.path.join(cabt_cg_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"native lib not found at {path} — is kaggle_environments installed?")
    return path


def download_cglib_dataset(dest):
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "kiyotah/cg-lib", "-p", dest, "--unzip"],
        check=True,
    )


def main():
    os.makedirs(OUT_CG, exist_ok=True)

    native_lib_path = find_kaggle_environments_native_lib()
    native_lib_name = os.path.basename(native_lib_path)

    tmp_download = os.path.join(OUT_DIR, "_cglib_download")
    api_py = os.path.join(tmp_download, "cg", "api.py")
    if not os.path.exists(api_py):
        os.makedirs(tmp_download, exist_ok=True)
        download_cglib_dataset(tmp_download)

    for fname in ("api.py", "sim.py", "utils.py", "game.py", "__init__.py"):
        src = os.path.join(tmp_download, "cg", fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT_CG, fname))

    shutil.copy(native_lib_path, os.path.join(OUT_CG, native_lib_name))

    print(f"Assembled local search-capable cg package at {OUT_CG}")
    print(f"Native lib: {native_lib_name} (from installed kaggle_environments)")
    print("Add this to sys.path to use: sys.path.insert(0, r'" + OUT_DIR + "')")

    sys.path.insert(0, OUT_DIR)
    from cg.api import all_card_data, all_attack
    cards = all_card_data()
    attacks = all_attack()
    print(f"Smoke test OK: {len(cards)} cards, {len(attacks)} attacks loaded via native lib.")


if __name__ == "__main__":
    main()
