#!/usr/bin/env bash
# Copy the committed state of this repository into a Code Ocean capsule and push it.
#
#   codeocean/sync_capsule.sh https://git.codeocean.com/capsule-xxxxxxx.git
#
# A capsule repository is not a mirror of this one: its root holds code/, data/, environment/,
# metadata/ and results/, and our tree belongs under code/. Only committed files are copied, via
# git archive, so an untracked checkpoint or a local image can never reach the capsule by accident.
set -euo pipefail

url="${1:?usage: sync_capsule.sh <capsule git url>}"
repo="$(cd "$(dirname "$0")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

echo "== cloning the capsule"
git clone "$url" "$work/capsule"           # username = your Code Ocean email, password = access token
cap="$work/capsule"
[ -d "$cap/code" ] || { echo "no code/ directory: is $url really a capsule?" >&2; exit 1; }

echo "== replacing code/ with the committed tree of $(git -C "$repo" rev-parse --short HEAD)"
find "$cap/code" -mindepth 1 -delete
git -C "$repo" archive HEAD | tar -x -C "$cap/code"
cp "$repo/codeocean/run" "$cap/code/run"
chmod +x "$cap/code/run" 2>/dev/null || true
git -C "$cap" update-index --add --chmod=+x code/run 2>/dev/null || true

echo "== pushing"
git -C "$cap" add -A
git -C "$cap" commit -q -m "sync from diganet $(git -C "$repo" rev-parse --short HEAD)" || { echo "nothing changed"; exit 0; }
git -C "$cap" push
echo "pushed; open the capsule and start a Reproducible Run"
