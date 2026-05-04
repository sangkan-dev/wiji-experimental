#!/bin/bash
# Reproduce all WIJI experiments sequentially
# Usage: bash scripts/reproduce_all.sh

set -e

echo "================================================"
echo "WIJI Phase 0 — Reproducing all experiments"
echo "================================================"
echo ""

# Check uv installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv not installed. Install with:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Sync dependencies
echo "[Setup] Syncing dependencies..."
uv sync

mkdir -p results

# Experiment 1
echo ""
echo "================================================"
echo "Experiment 1: Single Matrix"
echo "================================================"
uv run experiments/exp01_single_matrix.py 2>&1 | tee results/exp01_output.txt

# Experiment 2
echo ""
echo "================================================"
echo "Experiment 2: Multi-Layer Single Generator"
echo "================================================"
uv run experiments/exp02_multi_layer_single_gen.py 2>&1 | tee results/exp02_output.txt

# Experiment B4 (cliff edge — most informative)
echo ""
echo "================================================"
echo "Experiment B4: Cliff Edge Test"
echo "================================================"
uv run experiments/expB4_cliff_edge.py 2>&1 | tee results/expB4_output.txt

echo ""
echo "================================================"
echo "All experiments complete!"
echo "Results saved to: results/"
echo "================================================"
