#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="${REPO_ROOT}/patches/h1b_cellsam_source_rescue_20260319.patch"
TARGET_REPO="${REPO_ROOT}/cellSAM_source"
declare -A EXPECTED_SHA256=(
  ["cellSAM/sam_inference.py"]="90c5a95f5cf7b45e5e14f36d2d0b966001dd612c102c70cc169b28318363b6e8"
  ["cellSAM/AnchorDETR/models/anchor_detr.py"]="f467a66c1ed35efca7ca61b20355f8b25aa29a476d38255ed4b290cd503ebac8"
  ["cellSAM/AnchorDETR/models/transformer.py"]="1cf6e14e221fb0dfe995d66fbd38d1a730b40445522ad47592102d22678eb265"
  ["cellSAM/AnchorDETR/models/matcher.py"]="191ba95a57f84e744b93563d2540ae83abf416625b9f99f08f247989e0b639ce"
)

all_expected_hashes_match() {
  for rel in "${!EXPECTED_SHA256[@]}"; do
    local abs="${TARGET_REPO}/${rel}"
    if [[ ! -f "${abs}" ]]; then
      return 1
    fi
    local got
    got="$(sha256sum "${abs}" | awk '{print $1}')"
    if [[ "${got}" != "${EXPECTED_SHA256[$rel]}" ]]; then
      return 1
    fi
  done
  return 0
}

if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "Patch file not found: ${PATCH_FILE}" >&2
  exit 1
fi

if [[ ! -d "${TARGET_REPO}/.git" ]]; then
  echo "Target is not a git repo: ${TARGET_REPO}" >&2
  exit 1
fi

echo "[info] Applying patch to ${TARGET_REPO}"
if git -C "${TARGET_REPO}" apply --check "${PATCH_FILE}" >/dev/null 2>&1; then
  git -C "${TARGET_REPO}" apply "${PATCH_FILE}"
  echo "[ok] Patch applied."
elif git -C "${TARGET_REPO}" apply --reverse --check "${PATCH_FILE}" >/dev/null 2>&1; then
  echo "[ok] Patch already applied; nothing to do."
elif all_expected_hashes_match; then
  echo "[ok] Patch target hashes already match expected version; nothing to do."
else
  echo "[error] Patch check failed (neither apply nor reverse-check succeeded)." >&2
  exit 1
fi

echo "[next] verify with: git -C cellSAM_source status --short"
