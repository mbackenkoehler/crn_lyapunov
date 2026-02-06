#!/bin/bash

set -v
set -e

cd ~/crn_lyapunov

uv sync
uv pip install -e .

uv run python scripts/experiments.py
