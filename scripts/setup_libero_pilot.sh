#!/usr/bin/env bash
set -euo pipefail

# Keep the upstream runtime and its compatibility patches outside this repository.
runtime_root=${1:?Usage: scripts/setup_libero_pilot.sh /absolute/path/to/runtime}
mkdir -p "$runtime_root"
runtime_root=$(cd "$runtime_root" && pwd -P)

fetch_revision() {
    local repository=$1 revision=$2 directory=$3
    if [[ ! -e "$directory/.git" ]]; then
        git init --quiet "$directory"
        git -C "$directory" remote add origin "$repository"
    fi
    local current
    current=$(git -C "$directory" rev-parse --verify HEAD 2>/dev/null || true)
    if [[ -z "$current" ]]; then
        git -C "$directory" fetch --depth=1 origin "$revision"
        git -C "$directory" -c advice.detachedHead=false checkout --detach "$revision"
    elif [[ "$current" != "$revision" ]] || [[ -n "$(git -C "$directory" status --porcelain --untracked-files=no)" ]]; then
        printf 'Use a clean checkout at %s in %s or start with a new runtime path.\n' "$revision" "$directory" >&2
        return 1
    fi
}

openpi="$runtime_root/openpi"
libero="$runtime_root/LIBERO"
export GIT_LFS_SKIP_SMUDGE=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-$runtime_root/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$runtime_root/uv-python}"
fetch_revision https://github.com/Physical-Intelligence/openpi.git 215abfb217dbac7d5f1273282331b9b1866c0479 "$openpi"
(cd "$openpi" && uv sync --frozen --no-dev --python 3.11 --link-mode copy)
runtime_python="$openpi/.venv/bin/python"

# OpenPI's Torch backend expects its version-matched Transformers overlays.
"$runtime_python" - "$openpi" <<'PY'
from pathlib import Path
import shutil
import sys
import transformers

if transformers.__version__ != "4.53.2":
    raise RuntimeError(f"Expected Transformers 4.53.2, got {transformers.__version__}")
patch_dir = Path(sys.argv[1]) / "src/openpi/models_pytorch/transformers_replace"
shutil.copytree(patch_dir, Path(transformers.__file__).parent, dirs_exist_ok=True)
PY

fetch_revision https://github.com/Lifelong-Robot-Learning/LIBERO.git 8f1084e3132a39270c3a13ebe37270a43ece2a01 "$libero"
uv pip install --link-mode copy --python "$runtime_python" \
    'robosuite==1.4.1' 'mujoco==2.3.7' 'bddl==1.0.1' 'easydict==1.13' \
    'cloudpickle>=2.1' 'future>=0.18' 'matplotlib>=3.7' 'gym==0.26.2' \
    'numpy==1.26.4' 'torch==2.7.1' 'transformers==4.53.2' 'jax==0.5.3' 'flax==0.10.2'
uv pip install --link-mode copy --python "$runtime_python" --no-deps -e "$libero"
uv pip check --python "$runtime_python"

export LIBERO_CONFIG_PATH="$runtime_root/libero-config"
"$runtime_python" - "$libero" <<'PY'
import json
import os
from pathlib import Path
import sys

libero = Path(sys.argv[1])
root = libero / "libero/libero"
config = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
config.parent.mkdir(parents=True, exist_ok=True)
datasets = libero / "libero/datasets"
datasets.mkdir(exist_ok=True)
paths = {
    "benchmark_root": str(root),
    "bddl_files": str(root / "bddl_files"),
    "init_states": str(root / "init_files"),
    "assets": str(root / "assets"),
    "datasets": str(datasets),
}
contents = json.dumps(paths, indent=2) + "\n"
if config.exists() and config.read_text() != contents:
    raise RuntimeError(f"Existing LIBERO configuration differs: {config}")
config.write_text(contents)
PY

export MUJOCO_GL="${MUJOCO_GL:-egl}"
if [[ "$MUJOCO_GL" == egl || "$MUJOCO_GL" == osmesa ]]; then
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-$MUJOCO_GL}"
fi
"$runtime_python" - <<'PY'
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from openpi.training import config

assert benchmark.get_benchmark("libero_10")().n_tasks > 0
assert config.get_config("pi05_libero").model.pi05
print("OpenPI and LIBERO imports passed.")
PY
printf 'Pilot Python: %s\nLIBERO_CONFIG_PATH: %s\n' "$runtime_python" "$LIBERO_CONFIG_PATH"
