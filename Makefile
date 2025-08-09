include .env
export

PYTHON = python3
CONFIG = config/default.yaml
SRC_DIR = src

.PHONY: help data train evaluate test setup-hooks notebook clear-outputs clear-checkpoints clean

help:
	@echo "Usage:"
	@echo " make data              - Prepare dataset"
	@echo " make train             - Train model"
	@echo " make evaluate          - Evaluate model"
	@echo " make test              - Run tests"
	@echo " make setup-hooks       - Set up git pre-commit hook"
	@echo " make clear-outputs     - Clear all outputs"
	@echo " make clear-checkpoints - Clear all checkpoints"
	@echo " make clean             - Clean logs and models"

train:
	$(PYTHON) $(SRC_DIR)/train.py --config $(CONFIG)

evaluate:
	$(PYTHON) $(SRC_DIR)/evaluate.py --config $(CONFIG)

test:
	pytest tests/

setup-hooks:
	bash scripts/setup-hooks.sh
	@echo "✅ pre-commit hook has been successfully installed!"

clear-outputs:
	rm -rf outputs/*

clear-checkpoints:
ifeq ($(GIT_TRACK_CHECKPOINTS), YES)
	@echo "!! GIT_TRACK_CHECKPOINTS is set to YES; checkpoints won't be cleared."
else
	rm -rf models/checkpoints/*
	@echo ">> checkpoints have been cleared."
endif

clean:
	rm -rf outputs/* models/checkpoints/*