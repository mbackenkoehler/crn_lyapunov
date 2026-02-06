#!/bin/bash

set -v
set -e

cd ~/crn_lyapunov

uv sync
uv pip install -e .

uv run python scripts/experiments.py "bd" &
uv run python scripts/experiments.py "schloegl" &
uv run python scripts/experiments.py "parbd" &
uv run python scripts/experiments.py "comp" &
uv run python scripts/experiments.py "toggle" &
uv run python scripts/experiments.py "p53" &

wait
