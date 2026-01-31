SHELL := /bin/bash

PYTHON ?= python3
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

TASK ?= dock
STEPS ?= 200000
CKPT ?= artifacts/ppo_$(TASK).pt
LOG ?= artifacts/reward.csv
EPISODES ?= 10
DETERMINISTIC ?= 1

.PHONY: help venv deps train eval baseline plot clean

help:
	@echo "Targets:"
	@echo "  venv       Create virtual environment in $(VENV)"
	@echo "  deps       Install dependencies from requirements.txt"
	@echo "  train      Train PPO (TASK=$(TASK) STEPS=$(STEPS))"
	@echo "  eval       Run evaluation (TASK=$(TASK) CKPT=$(CKPT))"
	@echo "  baseline   Run naive controller (TASK=$(TASK) EPISODES=$(EPISODES))"
	@echo "  plot       Plot rewards (LOG=$(LOG))"
	@echo "  clean      Remove venv and artifacts"
	@echo ""
	@echo "Examples:"
	@echo "  make venv deps"
	@echo "  make train TASK=dock STEPS=200000"
	@echo "  make eval TASK=dock CKPT=artifacts/ppo_dock.pt DETERMINISTIC=1"
	@echo "  make baseline TASK=orbit EPISODES=5"
	@echo "  make plot LOG=artifacts/reward.csv"

venv:
	@test -d $(VENV) || $(PYTHON) -m venv $(VENV)

deps: venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

train: deps
	@$(PY) train.py --task $(TASK) --total-steps $(STEPS)

eval: deps
	@if [ "$(DETERMINISTIC)" = "1" ]; then \
		$(PY) evaluate.py --task $(TASK) --ckpt $(CKPT) --deterministic; \
	else \
		$(PY) evaluate.py --task $(TASK) --ckpt $(CKPT); \
	fi

baseline: deps
	@$(PY) baseline.py --task $(TASK) --episodes $(EPISODES)

plot: deps
	@$(PY) plot_rewards.py --log $(LOG)

clean:
	@rm -rf $(VENV) artifacts
