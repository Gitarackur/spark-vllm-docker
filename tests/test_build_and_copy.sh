#!/bin/bash
#
# Focused behavior tests for build-and-copy.sh image preparation.
# Uses fake docker/ssh/curl commands, so it never pulls images, builds images,
# copies to real hosts, or touches the repository wheel cache.

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TMP_BASE="$(mktemp -d)"
TEST_INDEX=0
TESTS_PASSED=0

cleanup() {
    rm -rf "$TMP_BASE"
}
trap cleanup EXIT

pass() {
    echo "[PASS] $1"
    TESTS_PASSED=$((TESTS_PASSED + 1))
}

fail() {
    echo "[FAIL] $1" >&2
    if [ -n "${OUTPUT_LOG:-}" ] && [ -f "$OUTPUT_LOG" ]; then
        echo "--- output ---" >&2
        sed -n '1,220p' "$OUTPUT_LOG" >&2
    fi
    if [ -n "${TEST_LOG:-}" ] && [ -f "$TEST_LOG" ]; then
        echo "--- command log ---" >&2
        sed -n '1,220p' "$TEST_LOG" >&2
    fi
    exit 1
}

setup_fixture() {
    TEST_INDEX=$((TEST_INDEX + 1))
    CASE_DIR="$TMP_BASE/case-$TEST_INDEX"
    FIXTURE_DIR="$CASE_DIR/project"
    FAKE_BIN_DIR="$CASE_DIR/bin"
    TEST_LOG="$CASE_DIR/commands.log"
    OUTPUT_LOG="$CASE_DIR/output.log"

    mkdir -p "$FIXTURE_DIR" "$FAKE_BIN_DIR"
    cp "$PROJECT_DIR/build-and-copy.sh" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/autodiscover.sh" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/Dockerfile" "$FIXTURE_DIR/"
    cp "$PROJECT_DIR/Dockerfile.mxfp4" "$FIXTURE_DIR/"
    mkdir -p "$FIXTURE_DIR/wheels"
    touch "$FIXTURE_DIR/wheels/flashinfer-test.whl"
    touch "$FIXTURE_DIR/wheels/vllm-test.whl"
    touch "$FIXTURE_DIR/test.env"
    : > "$TEST_LOG"
    : > "$OUTPUT_LOG"

    cat > "$FAKE_BIN_DIR/docker" <<'DOCKER'
#!/bin/bash
set -euo pipefail
echo "docker $*" >> "$TEST_LOG"
if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then
    echo "${LOCAL_IMAGE_ID:-sha256:local}"
    exit 0
fi
if [ "${1:-}" = "save" ]; then
    out=""
    while [ "$#" -gt 0 ]; do
        if [ "$1" = "-o" ]; then
            out="$2"
            shift 2
            continue
        fi
        shift
    done
    if [ -n "$out" ]; then
        printf 'fake image\n' > "$out"
    fi
fi
DOCKER

    cat > "$FAKE_BIN_DIR/ssh" <<'SSH'
#!/bin/bash
set -euo pipefail
echo "ssh $*" >> "$TEST_LOG"
target="${1:-}"
host="${target#*@}"
cmd="${*:2}"
if [[ "$cmd" == *"docker image inspect"* ]]; then
    case "$host" in
        samehost)
            echo "${LOCAL_IMAGE_ID:-sha256:local}"
            exit 0
            ;;
        diffhost)
            echo "sha256:remote"
            exit 0
            ;;
        *)
            exit 1
            ;;
    esac
fi
while IFS= read -r _line; do
    :
done
SSH

    cat > "$FAKE_BIN_DIR/curl" <<'CURL'
#!/bin/bash
set -euo pipefail
echo "curl $*" >> "$TEST_LOG"
exit 22
CURL

    chmod +x "$FAKE_BIN_DIR/docker" "$FAKE_BIN_DIR/ssh" "$FAKE_BIN_DIR/curl"
}

run_build() {
    (
        cd "$FIXTURE_DIR"
        PATH="$FAKE_BIN_DIR:$PATH" TEST_LOG="$TEST_LOG" ./build-and-copy.sh --config "$FIXTURE_DIR/test.env" "$@"
    ) > "$OUTPUT_LOG" 2>&1
}

assert_log_contains() {
    local pattern="$1"
    if ! grep -Eq "$pattern" "$TEST_LOG"; then
        fail "Expected command log to match: $pattern"
    fi
}

assert_log_not_contains() {
    local pattern="$1"
    if grep -Eq "$pattern" "$TEST_LOG"; then
        fail "Expected command log not to match: $pattern"
    fi
}

assert_output_contains() {
    local pattern="$1"
    if ! grep -Eq "$pattern" "$OUTPUT_LOG"; then
        fail "Expected output to match: $pattern"
    fi
}

test_default_uses_prebuilt() {
    setup_fixture
    run_build || fail "default run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm:latest vllm-node$'
    assert_log_not_contains '^docker build'
    pass "default pulls and tags prebuilt image"
}

test_tf5_uses_prebuilt_tf5_tag() {
    setup_fixture
    run_build --tf5 || fail "--tf5 run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm:latest vllm-node-tf5$'
    assert_log_not_contains '^docker build'
    pass "--tf5 pulls prebuilt image under vllm-node-tf5"
}

test_custom_tag_uses_prebuilt_custom_tag() {
    setup_fixture
    run_build -t custom-vllm || fail "custom tag run failed"
    assert_log_contains '^docker tag eugr/spark-vllm:latest custom-vllm$'
    assert_log_not_contains '^docker build'
    pass "custom tag pulls prebuilt image under requested tag"
}

test_default_gpu_arch_stays_prebuilt() {
    setup_fixture
    run_build --gpu-arch 12.1a || fail "default gpu arch run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build'
    pass "explicit default gpu arch still uses prebuilt image"
}

test_non_default_gpu_arch_uses_wheel_build() {
    setup_fixture
    run_build --gpu-arch 12.0f || fail "non-default gpu arch run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker build -t vllm-node '
    assert_log_contains 'NCCL_NVCC_GENCODE=-gencode=arch=compute_120,code=sm_120'
    pass "non-default gpu arch uses wheel build path"
}

test_use_wheels_uses_wheel_build() {
    setup_fixture
    run_build --use-wheels || fail "--use-wheels run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker build -t vllm-node '
    assert_log_contains 'NCCL_NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121'
    pass "--use-wheels preserves wheel build path with default NCCL gencode"
}

test_cleanup_stays_prebuilt() {
    setup_fixture
    run_build --cleanup || fail "--cleanup run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build'
    pass "--cleanup is orthogonal and still allows prebuilt path"
}

test_prebuilt_copy_parallel() {
    setup_fixture
    run_build -c host1,host2 --copy-parallel || fail "prebuilt copy run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_contains '^docker tag eugr/spark-vllm:latest vllm-node$'
    assert_log_contains '^docker save -o .* vllm-node$'
    assert_log_contains '^ssh .*@host1 docker load$'
    assert_log_contains '^ssh .*@host2 docker load$'
    pass "prebuilt path saves requested tag and supports parallel copy"
}

test_copy_skips_matching_remote_image() {
    setup_fixture
    run_build -c samehost || fail "matching remote copy run failed"
    assert_log_contains '^docker image inspect --format \{\{\.Id\}\} vllm-node$'
    assert_log_contains '^ssh .*@samehost docker image inspect --format '\''\{\{\.Id\}\}'\'' vllm-node$'
    assert_log_not_contains '^docker save '
    assert_log_not_contains '^ssh .*@samehost docker load$'
    assert_output_contains "Image 'vllm-node' is already up to date on .*@samehost; skipping\."
    assert_output_contains 'All remote images are up to date; skipping save/copy\.'
    pass "copy skips save/load when remote image ID matches local"
}

test_copy_only_updates_missing_or_different_hosts() {
    setup_fixture
    run_build -c samehost,host1 --copy-parallel || fail "mixed remote copy run failed"
    assert_log_contains '^docker save -o .* vllm-node$'
    assert_log_not_contains '^ssh .*@samehost docker load$'
    assert_log_contains '^ssh .*@host1 docker load$'
    pass "copy loads only hosts whose image ID is missing or different"
}

test_no_build_skips_prebuilt() {
    setup_fixture
    run_build --no-build -c host1 || fail "--no-build copy run failed"
    assert_log_not_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker tag eugr/spark-vllm:latest'
    assert_log_contains '^docker save -o .* vllm-node$'
    assert_log_contains '^ssh .*@host1 docker load$'
    pass "--no-build skips prebuilt pull and copies existing local tag"
}

test_build_only_flags_warn_on_prebuilt() {
    setup_fixture
    run_build --network host --full-log -j 4 || fail "build-only flags prebuilt run failed"
    assert_log_contains '^docker pull eugr/spark-vllm:latest$'
    assert_log_not_contains '^docker build'
    assert_output_contains 'Warning: --network is only used for Docker builds; ignoring it while pulling eugr/spark-vllm:latest\.'
    assert_output_contains 'Warning: --full-log is only used for Docker builds; ignoring it while pulling eugr/spark-vllm:latest\.'
    assert_output_contains 'Warning: --build-jobs is only used for Docker builds; ignoring it while pulling eugr/spark-vllm:latest\.'
    pass "build-only flags warn but do not force wheel path"
}

test_default_uses_prebuilt
test_tf5_uses_prebuilt_tf5_tag
test_custom_tag_uses_prebuilt_custom_tag
test_default_gpu_arch_stays_prebuilt
test_non_default_gpu_arch_uses_wheel_build
test_use_wheels_uses_wheel_build
test_cleanup_stays_prebuilt
test_prebuilt_copy_parallel
test_copy_skips_matching_remote_image
test_copy_only_updates_missing_or_different_hosts
test_no_build_skips_prebuilt
test_build_only_flags_warn_on_prebuilt

echo "Passed $TESTS_PASSED build-and-copy tests."
