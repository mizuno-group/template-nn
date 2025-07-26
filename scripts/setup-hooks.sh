#!/bin/bash

# copy pre-commit hook script to the .git/hooks directory
HOOK_SOURCE=$(git rev-parse --show-toplevel)/scripts/pre-commit-check.sh
HOOK_DESTINATION=$(git rev-parse --show-toplevel)/.git/hooks/pre-commit

cp "$HOOK_SOURCE" "$HOOK_DESTINATION"
chmod +x "$HOOK_DESTINATION"