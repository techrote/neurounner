#!/usr/bin/env bash
# Dedicated source-built native research harness. Actual hand-authored proposals
# only; not model completions, not shared TRAIN admission or a candidate freeze.
set -euo pipefail
test -n "${RUNNER_TEMP:-}"
test -n "${GITHUB_SHA:-}"
test "${RUNNER_ENVIRONMENT:-}" = github-hosted
mkdir -p evidence/build
git rev-parse HEAD > evidence/build/source-commit.txt
git rev-parse 'HEAD^{tree}' > evidence/build/source-tree.txt
test "$(git rev-parse HEAD)" = "$GITHUB_SHA"
cp experiments/agent_painter/lanes/lfm25_vl/contrastive-native-request.json evidence/build/request.json
python3 tools/ci/check_ci_provider_policy.py > evidence/build/ci-provider.json

git lfs install --local
git lfs pull --include='godot/addons/godot-rapier2d/bin/libgodot_rapier.linux.x86_64-unknown-linux-gnu.so' --exclude=''
echo '566e68ae53bfb2eb7018c8e7d0dd3727121bf696462039b780a33d6b8369c66a  godot/addons/godot-rapier2d/bin/libgodot_rapier.linux.x86_64-unknown-linux-gnu.so' | sha256sum -c --strict

python3 -m venv "$RUNNER_TEMP/lfm-native"
"$RUNNER_TEMP/lfm-native/bin/python" -m pip install --require-hashes -r tools/requirements-build.txt > evidence/build/install-scons.log 2>&1
"$RUNNER_TEMP/lfm-native/bin/python" -m pip install 'Pillow==13.0.0' > evidence/build/install-pillow.log 2>&1
export GODOT_CPP_ROOT="$RUNNER_TEMP/lfm-godot-cpp"
export SCONS_PYTHON="$RUNNER_TEMP/lfm-native/bin/python"
export SCONS_JOBS=4
export CYBERSAND_REAL_CXX=/usr/bin/g++-13
export LINUX_CXX=/usr/bin/g++-13
export CYBERSAND_NATIVE_OUTPUT="$RUNNER_TEMP/lfm-native.so"
mkdir -p "$GODOT_CPP_ROOT"
git -C "$GODOT_CPP_ROOT" init
git -C "$GODOT_CPP_ROOT" remote add origin https://github.com/godotengine/godot-cpp.git
timeout 600 git -C "$GODOT_CPP_ROOT" fetch --depth=1 origin 101ae38034304346a46ea9ea84ae156d3e860496 > evidence/build/godot-cpp-fetch.log 2>&1
git -C "$GODOT_CPP_ROOT" checkout --detach FETCH_HEAD
if ! timeout 3600 bash tools/build_pinned_godot_cpp.sh linux > evidence/build/godot-cpp.log 2>&1; then tail -120 evidence/build/godot-cpp.log; exit 1; fi
if ! timeout 3600 bash tools/build_native_extension.sh > evidence/build/native.log 2>&1; then tail -120 evidence/build/native.log; exit 1; fi
cp "$CYBERSAND_NATIVE_OUTPUT" godot/addons/cybersand_native/bin/libcybersand_native.linux.x86_64.so
mkdir -p evidence/build/runtime
cp "$CYBERSAND_NATIVE_OUTPUT" evidence/build/runtime/libcybersand_native.linux.x86_64.so
bash tools/check_linux_runtime_floor.sh > evidence/build/abi-floor.log 2>&1
python3 experiments/agent_painter/native/runtime.py stamp --source-sha "$GITHUB_SHA" --output evidence/native-build.json

timeout 600 curl -fsSL --retry 3 -o "$RUNNER_TEMP/lfm-godot.zip" 'https://downloads.godotengine.org/?flavor=stable&platform=linux.64&slug=linux.x86_64.zip&version=4.7'
echo "0b1a6c54c2c619c12e169fe9241edda4b81080b519451cec2984bf0d2c6cb73c  $RUNNER_TEMP/lfm-godot.zip" | sha256sum -c --strict
mkdir -p "$RUNNER_TEMP/lfm-godot"
unzip -q "$RUNNER_TEMP/lfm-godot.zip" -d "$RUNNER_TEMP/lfm-godot"
export GODOT_BIN="$RUNNER_TEMP/lfm-godot/Godot_v4.7-stable_linux.x86_64"
chmod +x "$GODOT_BIN"
"$GODOT_BIN" --version > evidence/build/godot-version.txt
if ! timeout 1200 "$GODOT_BIN" --headless --path godot --editor --quit > evidence/build/import.log 2>&1; then tail -120 evidence/build/import.log; exit 1; fi
"$RUNNER_TEMP/lfm-native/bin/python" experiments/agent_painter/lanes/lfm25_vl/contrastive_native.py \
  --proposals experiments/agent_painter/lanes/lfm25_vl/contrastive_proposals_v1.json \
  --disposition experiments/agent_painter/lanes/lfm25_vl/contrastive_disposition.json \
  --corpus evidence/original-native/corpus \
  --godot "$GODOT_BIN" --runtime-manifest evidence/native-build.json \
  --source-sha "$GITHUB_SHA" --output evidence/lfm-contrastive-native
"$RUNNER_TEMP/lfm-native/bin/python" experiments/agent_painter/qualification.py regressions --godot "$GODOT_BIN" --evidence evidence/regressions
