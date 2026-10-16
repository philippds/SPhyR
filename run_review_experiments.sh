#!/usr/bin/env bash
# Hosted-model runs for the PhysWorldAI camera-ready. Run from the SPhyR repo root.
# Every run resumes from what is already stored, so it is safe to re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")" 2>/dev/null || true

SUBJECTS="10_random_cell_easy 10_random_cell_hard 3_random_row_easy 3_random_row_hard full_easy full_hard"
MODELS="anthropic/claude-opus-5 google/gemini-3.8-flash"

# 1. Sparse answer format, both grid sizes, on exactly the samples of the stored runs (2 x 2 x 600 calls).
for V in v2-20 v2; do
  uv run python -m sphyr.run_eval main --models $MODELS --dataset-version "$V" \
    --replay-from "anthropic_claude-opus-5_$V" --prompt-style sparse \
    --subjects $SUBJECTS --concurrency 8
done

# 2. Gemini grid format again, now with finish_reason/token logging (200 calls):
#    3_random_row_easy is the 41-point example; full_hard puts Gemini on Opus's samples
#    (the stored Gemini v2-20 full_hard run shares only 34 of 100 samples with Opus).
uv run python -m sphyr.run_eval main --models google/gemini-3.8-flash --dataset-version v2-20 \
  --replay-from anthropic_claude-opus-5_v2-20 --subjects 3_random_row_easy full_hard \
  --name-suffix _rerun --concurrency 8

# 3. Tables and diagnostics for the paper.
uv run python -m sphyr.analyze_results formats --latex --runs \
  anthropic_claude-opus-5_v2 google_gemini-3.8-flash_v2 \
  anthropic_claude-opus-5_v2-20 google_gemini-3.8-flash_v2-20 \
  google_gemini-3.8-flash_rerun_v2-20:google_gemini-3.8-flash_sparse_v2-20
uv run python -m sphyr.analyze_results invalid --runs google_gemini-3.8-flash_rerun_v2-20
