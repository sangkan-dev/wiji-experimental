<div align="center">

# WIJI — Weight-Implicit Just-in-time Inference

**ꦮꦶꦗꦶ** *— biji, benih, asal mula*

*An open research project that asked a hard question and got an honest answer.*

[![Status](https://img.shields.io/badge/status-phase%200%20closed-blue)](.)
[![Result](https://img.shields.io/badge/result-negative%20(but%20valuable)-orange)](.)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

---

## TL;DR

WIJI adalah eksperimen riset terbuka yang mempertanyakan asumsi fundamental di deep learning: **apakah weight neural network harus disimpan sebagai matriks angka statis?**

Setelah **9 eksperimen sistematik** dengan **5 arsitektur generator yang berbeda**, jawabannya adalah: **ya, untuk model yang sudah dilatih.**

Tapi journey ini menghasilkan **bukti empiris yang valuable** tentang batas fundamental coordinate-based weight reconstruction. Phase 0 ditutup dengan negative results yang well-documented dan **terbuka untuk siapapun yang mau melanjutkan ke arah berbeda** (co-training, hypernetwork training from scratch).

## Pertanyaan Penelitian

Bisakah weight LLM yang sudah dilatih (TinyLlama 1.1B) di-rekonstruksi dari generator coordinate-based yang jauh lebih kecil, sehingga model bisa berjalan dengan footprint memori minimal?

**Hipotesis awal**: Weight punya redundansi yang bisa di-exploit oleh generator dengan parameter signifikan lebih sedikit.

**Hasil**: Hipotesis terbukti benar untuk **single layer**, tapi **gagal untuk multi-layer** karena **floor of irreducible noise** di weight yang dihasilkan SGD training.

## Hasil Eksperimen — The Full Story

| # | Eksperimen | Arsitektur | Compression | Output Quality |
|---|-----------|------------|-------------|----------------|
| 1 | Single matrix swap | Coordinate MLP | 25x | ✅ Functional |
| 2 | Multi-layer single gen | Coordinate MLP + layer emb | 545x | ❌ Gibberish |
| A | Full attention layer 0 | Coordinate MLP + comp emb | 56x | ✅ Functional |
| B2 | Per-layer generator | 22 × Coordinate MLP | 25x | ❌ Gibberish |
| B3 | Adaptive capacity | 22 × Coordinate MLP (varied) | 15x | ❌ Worse |
| B4 | Cliff edge test | (uses B3 generators) | varies | 🔍 Cliff at N=2 |
| C1 | **Fourier Features** | Fourier encoding + MLP | 42x | ❌ Same plateau |
| D1 | **FFN target** | Fourier + dynamic | 116x | ❌ Same plateau |
| E1 | **SIREN** | Sinusoidal activation | 28x | ❌ Same plateau |
| F1 | **Mixture of Generators** | 16-expert sharding | 31x | ❌ Same plateau |

**The Smoking Gun**:

```
Final MSE Layer 21 across 5 different architectures:
  Baseline (B2) :  0.000373
  Adaptive (B3) :  0.000366
  Fourier  (C1) :  0.000356
  SIREN    (E1) :  0.000371
  MoG      (F1) :  0.000377
```

5 arsitektur yang fundamentally berbeda — coordinate MLP, Fourier features (NeRF-style), SIREN (sinusoidal), Mixture of Generators (sharding) — semuanya **mentok di MSE plateau yang hampir identik**.

Ini bukti yang sangat kuat bahwa **batas yang kami temui bukan batas arsitektur generator**, tapi **batas fundamental dari weight yang dihasilkan SGD training**.

## Insight Utama

### 1. MSE plateau adalah floor, bukan ceiling

Weight neural network setelah training **bukan smooth function dari koordinat**. Itu **stochastic outcome dari proses optimisasi** — dependent pada random init, batch order, dan ribuan microevents selama training.

**Tidak ada generator coordinate-based** yang bisa "compute" nilai weight dari koordinat saja, karena nilai itu **bukan fungsi dari koordinat**. Itu **artifact dari sejarah training**.

Ini insight yang signifikan. Banyak research di implicit neural representation berasumsi bahwa learned weight punya struktur yang bisa di-fit oleh fungsi smooth. Eksperimen kami **mempertanyakan asumsi itu** secara empiris.

### 2. Cliff edge phenomenon

Model collapse **tidak gradual** seiring jumlah layer di-swap. Ada phase transition tajam antara N=1 dan N=3.

Yang menarik: N=22 (semua layer di-swap) menghasilkan output meski gibberish, sementara N=5-16 menghasilkan empty string. **Internal consistency lebih penting dari absolute correctness** dalam multi-component systems.

### 3. MSE Loss bukan reliable predictor untuk LLM quality

Diagnostik A: MSE 6x lebih tinggi dari Eksperimen 1, output **tetap berkualitas tinggi**. Eksperimen 2: MSE hanya 3x lebih tinggi, output **collapse total**.

**Implikasi**: Research di area implicit weight representation perlu metric yang lebih baik dari MSE — output KL-divergence, perplexity comparison, atau task-specific accuracy.

## Status Project

🔴 **Phase 0 — CLOSED with negative results**

Setelah 9 eksperimen sistematik, kami **tidak bisa demonstrate** bahwa weight LLM yang sudah dilatih bisa di-compress lebih dari N=1 layer dengan coordinate-based generator. Ini bukan kegagalan komunikasi atau eksekusi — ini **fundamental limit** yang teridentifikasi melalui eksperimen yang hati-hati.

### Apa yang Mungkin (untuk orang lain)

Jalur yang **belum kami eksplor** dan mungkin menghasilkan hasil berbeda:

1. **Co-Training / HyperNetwork Training from Scratch**: Train model from scratch dimana generator adalah part of the architecture. Hypothesis: weight yang **terbentuk** dari generator akan compressible by definition. Sudah ada di literature (Ha et al. 2016+) tapi belum ada yang push ke LLM-scale.

2. **Output-aware loss**: Daripada minimize MSE pada weight, minimize KL-divergence pada output distribusi. Computationally expensive tapi might bypass MSE plateau.

3. **Different model architectures**: Test pada Mamba/SSM yang struktur weightnya mungkin berbeda dari transformer.

4. **Distillation approach**: Train small student dengan weight generated from generator, distill from teacher. Mengubah problem dari "fit existing weight" ke "learn capabilities."

## Struktur Repository

```
wiji-experimental/
├── README.md                  # File ini
├── LICENSE                    # MIT
├── CONTRIBUTING.md            # Panduan kontribusi
├── pyproject.toml
│
├── docs/
│   ├── PRD.md                 # Product Requirements Document (historical)
│   ├── RESULTS.md             # Hasil lengkap 9 eksperimen
│   ├── INSIGHTS.md            # Pelajaran teknis dan filosofis
│   ├── POSTMORTEM.md          # Analisis kenapa Phase 0 ditutup
│   ├── RESEARCH-FOUNDATIONS.md  # Paper-paper pendukung
│   └── CHANGELOG.md
│
├── experiments/
│   ├── exp01_single_matrix.py
│   ├── exp02_multi_layer_single_gen.py
│   ├── expA_full_attention_layer0.py
│   ├── expB2_per_layer_generator.py
│   ├── expB3_adaptive_capacity.py
│   ├── expB4_cliff_edge.py
│   ├── expC1_fourier_features.py     # Negative result
│   ├── expD1_ffn_target.py            # Negative result
│   ├── expE1_siren_o_proj.py          # Negative result
│   └── expF1_mixture_of_generators.py # Negative result
│
├── results/                   # Raw outputs from all experiments
├── scripts/
└── publications/              # LinkedIn, Medium, paper strategy
```

## Quick Start

```bash
git clone https://github.com/sangkan-dev/wiji-experimental.git
cd wiji-experimental
uv sync

# Run any experiment
uv run experiments/exp01_single_matrix.py
```

**Hardware**: 16 GB RAM laptop, no GPU required.

## Contributing

WIJI Phase 0 ditutup, tapi repo terbuka untuk:

- **Reproduce results** dan validasi temuan kami
- **Critique methodology** — apakah ada flaw di experimental setup yang membuat hasil misleading?
- **Try alternative approaches** (co-training, output-aware loss, dll) sebagai fork atau Phase 1
- **Discuss philosophy** — apakah kesimpulan kami terlalu pesimistis?

Lihat [CONTRIBUTING.md](CONTRIBUTING.md).

## Filosofi Project

WIJI dimulai dari pertanyaan yang ambisius dan ditutup dengan jawaban yang jujur. Bagi saya, itu adalah research yang valid — bahkan yang valuable.

Banyak project AI yang dimulai dengan klaim besar dan tidak pernah memberikan bukti. WIJI berani mengatakan: *"Ini yang kami coba. Ini yang kami pelajari. Ini batas yang kami temui."*

Negative results yang well-documented adalah **kontribusi nyata** ke komunitas riset. Setidaknya orang berikutnya yang punya ide serupa akan tahu apa yang sudah dicoba dan kenapa itu tidak bekerja — menghemat berbulan-bulan effort.

> *"Mari kita lebih menggila di dunia yang udah gila ini."*

Project ini gila. Hasilnya jujur. Itu cukup.

## Lisensi

MIT License — bebas untuk dipakai, modifikasi, dan extend.

## Acknowledgments

- **Claude (Anthropic)** untuk research collaboration intens
- **Microsoft Research** untuk BitNet
- **Komunitas hypernetwork research** (D'OH, fractal generative models, etc.)
- **Filosofi Jawa** untuk naming dan way-of-thinking
- **Komunitas yang bersedia membaca negative results** dengan respect

---

<div align="center">

**ꦮꦶꦗꦶ** — biji ditanam, kebenaran tumbuh.

*Sometimes the most valuable research output is knowing what doesn't work.*

Built with curiosity by [Sangkan](https://github.com/sangkan-dev) · 2026

</div>