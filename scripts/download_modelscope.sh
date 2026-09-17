#!/bin/bash
# Download a ModelScope model repo with plain curl (no `modelscope` / git-lfs needed).
# SoulChat2.0 checkpoints are only published on ModelScope.
#
# Usage: scripts/download_modelscope.sh YIRONGCHEN/SoulChat2.0-Qwen2-7B ft_models/SoulChat2.0-Qwen2-7B
set -u
MODEL=${1:?"usage: $0 <org/model> <dest_dir>"}
DEST=${2:?"usage: $0 <org/model> <dest_dir>"}
mkdir -p "$DEST"
FILES=$(curl -s "https://www.modelscope.cn/api/v1/models/$MODEL/repo/files?Revision=master&Recursive=true" \
  | python -c "import sys,json; [print(f['Path']) for f in json.load(sys.stdin)['Data']['Files'] if f['Type']!='tree']")
for f in $FILES; do
  echo "[$(date +%H:%M:%S)] $f"
  mkdir -p "$DEST/$(dirname "$f")"
  curl -sL -C - --retry 5 --retry-delay 10 -o "$DEST/$f" \
    "https://www.modelscope.cn/api/v1/models/$MODEL/repo?Revision=master&FilePath=$f" || echo "FAILED $f"
done
echo "[$(date +%H:%M:%S)] DONE"
ls -la "$DEST"
