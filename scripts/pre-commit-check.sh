#!/bin/bash

# .envがある場合のみ読み込む
if [ -f "$(git rev-parse --show-toplevel)/.env" ]; then
    export $(grep -v '^#' "$(git rev-parse --show-toplevel)/.env" | xargs)
fi

# デフォルトの上限を1GBに設定
DEFAULT_MAX_SIZE_GB=1
maxsize_gb=${PRE_COMMIT_FILE_SIZE_GB:-$DEFAULT_MAX_SIZE_GB}
maxsize_bytes=$((maxsize_gb * 1024 * 1024 * 1024))

files=$(git diff --cached --name-only)

for file in $files
do
    if [ -f "$file" ]; then
        filesize=$(stat -c%s "$file")
        if [ "$filesize" -gt "$maxsize_bytes" ]; then
            echo "!! Error: \"$file\" exceeds the maximum size of ${maxsize_gb}GB."
            echo "Current size is $(echo "scale=2; $filesize/1024/1024/1024" | bc) GB."
            echo "Commit aborted. Please reduce the file size or exclude this file from commit."
            exit 1
        fi
    fi
done

exit 0