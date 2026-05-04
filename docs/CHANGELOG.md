# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-05-03

### Added — Phase 0 Validation

- Initial repository structure
- PRD (Product Requirements Document) for WIJI concept
- 5 systematic experiments validating WIJI Phase 0 hypothesis
- Documentation: README, RESULTS, INSIGHTS, RESEARCH-FOUNDATIONS
- Publication materials: LinkedIn drafts, Medium article
- Reproduce all script

### Experiments Conducted

- `exp01_single_matrix.py` — Single matrix reconstruction (✅ Success, 25x)
- `exp02_multi_layer_single_gen.py` — Multi-layer single generator (❌ Failure)
- `expA_full_attention_layer0.py` — Full attention layer 0 (✅ Success, 56x)
- `expB2_per_layer_generator.py` — Per-layer generator (❌ Failure)
- `expB3_adaptive_capacity.py` — Adaptive capacity (❌ Failure)
- `expB4_cliff_edge.py` — Progressive layer swap (🔍 Cliff edge found at N=2)

### Key Findings

- Validated that single-layer weight compression up to 56x preserves output
- Discovered cliff edge phenomenon at N=2 layer swaps
- Empirically demonstrated MSE Loss is not a reliable predictor for LLM quality
- Identified spectral bias as fundamental limitation of coordinate MLPs

## [Planned 0.2.0] - Future

### To Add — Phase 1 Experiments

- Fourier features encoding (NeRF-style)
- SIREN activation experiment
- Output-aware loss (KL divergence)
- FFN layer reconstruction tests
- Tests on different model sizes (Qwen 1.5B, Phi-3 mini)
- Streaming inference system prototype

### To Add — Documentation

- ArXiv paper draft
- Tutorial blog posts
- Video walkthrough
- Indonesian translation of all docs
