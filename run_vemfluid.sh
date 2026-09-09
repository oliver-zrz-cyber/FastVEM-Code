#!/usr/bin/env bash
set -euo pipefail

# Resolve paths from this script so it can be launched from any directory.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT_DIR}/cmake-build-relwithdebinfo"
EXE="${BUILD_DIR}/VEMFLUID"
MESH="${ROOT_DIR}/models/ball.obj"
SOLVER="gmg"
RESOLUTION=64

usage() {
    cat <<'EOF'
Usage: bash run_vemfluid.sh [--solver gmg|amg] [--resolution N]

  --solver      Pressure solver: gmg (default) or amg.
  --resolution  Finest grid resolution; default: 64.
                GMG accepts 8, 16, 32, 64, ... .
  -h, --help    Show this help.
EOF
}

while (($#)); do
    case "$1" in
        --solver)
            (($# >= 2)) || { echo "--solver requires gmg or amg" >&2; exit 2; }
            SOLVER="$2"
            shift 2
            ;;
        --resolution)
            (($# >= 2)) || { echo "--resolution requires an integer" >&2; exit 2; }
            RESOLUTION="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "${SOLVER}" in
    gmg|amg) ;;
    *) echo "Unknown solver: ${SOLVER} (use gmg or amg)" >&2; exit 2 ;;
esac
if ! [[ "${RESOLUTION}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Resolution must be a positive integer: ${RESOLUTION}" >&2
    exit 2
fi

OUTPUT_DIR="${ROOT_DIR}/outputs/ball-${SOLVER}-${RESOLUTION}"

if [[ ! -f "${MESH}" ]]; then
    echo "Input mesh not found: ${MESH}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_DIR}"

# Configure and build before every run so the executable matches the source tree.
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build "${BUILD_DIR}" --parallel

if [[ ! -x "${EXE}" ]]; then
    echo "Build completed but executable was not produced: ${EXE}" >&2
    exit 1
fi

# Keep the legacy argument positions expected by main.cpp.
ARGS=(
    "-"                  # 1: unused legacy argument
    "ball-${SOLVER}-${RESOLUTION}" # 2: output filename prefix
    "0.5" "0.1" "0.5"   # 3-5: smoke source center (x, y, z)
    "0.08"               # 6: smoke source radius
    "${OUTPUT_DIR}/"     # 7: output directory (trailing slash required)
    "1"                  # 8: CFL mode
    "-"                  # 9: unused legacy AABB path
    "${MESH}"            # 10: boundary mesh
    "--solver" "${SOLVER}"
    "--resolution" "${RESOLUTION}"
)

# Keep generated meshes, tracer frames, and profiling data together.
cd "${OUTPUT_DIR}"
export VEMFLUID_PROFILE_CSV="${OUTPUT_DIR}/function_memory_profile.csv"
exec "${EXE}" "${ARGS[@]}"
