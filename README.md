<div align="center">

# WIJI — Weight-Implicit Just-in-time Inference

**ꦮꦶꦗꦶ** *— biji, benih, asal mula*

*Memaksimalkan yang minimal — dari biji menjadi hutan kecerdasan.*

[![Status](https://img.shields.io/badge/status-experimental-orange)](.)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Phase](https://img.shields.io/badge/phase-0%20validation-yellow)](.)

</div>

---

## Apa itu WIJI?

WIJI adalah eksperimen riset terbuka untuk mempertanyakan asumsi fundamental yang ada di seluruh deep learning sejak 1986:

> *"Apakah pengetahuan jaringan neural HARUS disimpan sebagai weight matriks statis?"*

Konsep WIJI: **weight tidak disimpan, weight ditumbuhkan dari biji**. Sebuah generator kecil dengan parameter minimal di-train untuk merekonstruksi weight model besar saat dibutuhkan, mirip seperti DNA yang menumbuhkan pohon dari biji.

**Goal**: AI model 30B+ jalan di smartphone tanpa cloud, dengan footprint memori beberapa puluh MB.

## Status Saat Ini

🟡 **Phase 0 — Validation in progress**

Kami sedang menguji apakah weight model existing (TinyLlama 1.1B) bisa di-rekonstruksi dengan loss yang acceptable dari generator kecil. Hasil sejauh ini: **partial success** — single layer compression 56x bekerja, multi-layer compression menemui cliff edge di N=2.

Lihat [RESULTS.md](docs/RESULTS.md) untuk dokumentasi lengkap eksperimen.

## Hasil Eksperimen Terbaru

| # | Eksperimen | Compression | Hasil |
|---|-----------|-------------|-------|
| 1 | Single matrix swap (`o_proj` layer 0) | 25x | ✅ Functional output |
| 2 | Multi-layer single generator (22 layers) | 545x | ❌ Gibberish output |
| A | Full attention layer 0 (Q/K/V/O) | 56x | ✅ Functional output |
| B2 | Per-layer generator (22 layers) | 25x | ❌ Gibberish output |
| B3 | Adaptive capacity per-layer | 15x | ❌ Worse output |
| B4 | Cliff edge test (progressive swap) | varies | 🔍 Cliff edge found at N=2 |

## Insight Kunci yang Sudah Ditemukan

1. **MSE Loss BUKAN reliable predictor** untuk output quality LLM (terbukti empiris)
2. **Cliff edge tajam di N=2** — model collapse setelah lebih dari 1 layer di-replace
3. **Capacity tidak menyelesaikan masalah** — generator 10x lebih besar tetap plateau di MSE 0.0003-0.0004
4. **Spectral bias hipotesis** — coordinate MLP standar mungkin punya hard limit untuk fitting weight transformer
5. **N=22 lebih baik dari N=5-16** — internal consistency matters more than absolute error

## Riset Pendukung

WIJI dibangun di atas beberapa breakthrough riset terbaru:

- [D'OH: Decoder-Only Random Hypernetworks](https://arxiv.org/abs/2403.19163) (ACCV 2024)
- [Recursive Self-Similarity in Deep Weight Spaces](https://arxiv.org/abs/2503.14298) (Mar 2025)
- [BitNet b1.58](https://arxiv.org/abs/2402.17764) (Microsoft 2024)
- [Pre-Attention Expert Prediction for MoE](https://arxiv.org/abs/2511.10676) (Nov 2025)
- [Fractal Generative Models](https://arxiv.org/abs/2502.17437) (Feb 2025)

Lihat [docs/RESEARCH-FOUNDATIONS.md](docs/RESEARCH-FOUNDATIONS.md) untuk konteks lengkap.

## Struktur Repository

```
wiji-experimental/
├── README.md              # File ini
├── LICENSE                # MIT License
├── pyproject.toml         # Python dependencies (uv)
├── docs/
│   ├── PRD.md             # Product Requirements Document lengkap
│   ├── RESULTS.md         # Hasil semua eksperimen + analisis
│   ├── INSIGHTS.md        # Pelajaran teknis dari journey ini
│   ├── RESEARCH-FOUNDATIONS.md  # Paper-paper yang menjadi fondasi
│   └── CHANGELOG.md       # Log perubahan
├── experiments/
│   ├── exp01_single_matrix.py
│   ├── exp02_multi_layer_single_gen.py
│   ├── expA_full_attention_layer0.py
│   ├── expB2_per_layer_generator.py
│   ├── expB3_adaptive_capacity.py
│   └── expB4_cliff_edge.py
├── results/
│   ├── exp01_output.txt
│   ├── exp02_output.txt
│   ├── ...
│   └── compiled_results.csv
└── scripts/
    └── reproduce_all.sh
```

## Quick Start

```bash
# Clone repo
git clone https://github.com/sangkan-dev/wiji-experimental.git
cd wiji-experimental

# Setup environment dengan uv
uv sync

# Run eksperimen pertama (single matrix)
uv run experiments/exp01_single_matrix.py

# Atau run semua eksperimen
bash scripts/reproduce_all.sh
```

**Hardware requirement minimal:**
- 16 GB RAM
- CPU 4 core (GPU tidak diperlukan)
- ~5 GB disk space (untuk model TinyLlama)

## Roadmap Berikutnya

### Short term (1-2 minggu)
- [ ] Test eksperimen dengan Fourier Features encoding
- [ ] Test eksperimen pada FFN layer (`gate_proj`, `up_proj`, `down_proj`)
- [ ] Test eksperimen pada model lebih besar (Qwen 1.5B)
- [ ] Output-aware loss function (KL-divergence)

### Medium term (1-2 bulan)
- [ ] Streaming inference system (Jalur C — only 1 layer in RAM)
- [ ] Fractal hypernetwork generator
- [ ] Rust port untuk performance benchmarking

### Long term (3-6 bulan)
- [ ] Mobile deployment (Android/iOS)
- [ ] Submit paper ke MLSys / NeurIPS Workshop
- [ ] Open source release v1.0

## Filosofi Project

WIJI bukan tentang ngalahin Claude atau GPT-4. WIJI tentang **decentralisasi kecerdasan**.

Kalau tren AI sentralisasi diteruskan, masa depan adalah dystopia dimana hanya 5 perusahaan di dunia yang punya AI kelas atas, dan sisanya menyewa. WIJI adalah taruhan melawan tren itu — taruhan bahwa kecerdasan bisa dibuat *minimal*, *terjangkau*, dan *milik semua orang*.

> *"Mari kita jadi lebih gila, mari kita lebih menggila di dunia yang udah gila ini."*

## Kontribusi

Project ini sangat eksperimental dan terbuka untuk siapapun yang tertarik. Beberapa cara berkontribusi:

- **Reproduce eksperimen** dan laporkan hasil di issues
- **Coba eksperimen baru** dengan model atau dataset berbeda
- **Bantu validasi hipotesis** dengan literature search
- **Kritik metodologi** — sangat dibutuhkan untuk rigor ilmiah

Lihat [CONTRIBUTING.md](CONTRIBUTING.md) untuk detail.

## Diskusi

- 💬 [GitHub Discussions](https://github.com/sangkan-dev/wiji-experimental/discussions)
<!-- - 🐦 Twitter/X: [@hasanh47](https://twitter.com/hasanh47) -->
- 📝 LinkedIn: [HasanH47](https://linkedin.com/in/hasanh47)

## Lisensi

MIT License — bebas untuk dipakai, modifikasi, dan distribusi. Lihat [LICENSE](LICENSE).

## Acknowledgments

- **Claude (Anthropic)** untuk research collaboration intens dalam mengeksplorasi konsep ini
- **Microsoft Research** untuk BitNet yang membuka pintu thinking di luar FP16
- **Komunitas hypernetwork research** yang sudah meletakkan fondasi matematis
- **Filosofi Jawa** yang menginspirasi naming dan way-of-thinking project ini

---

<div align="center">

**ꦮꦶꦗꦶ** — biji ditanam, hutan tumbuh.

Built with curiosity by [Sangkan](https://github.com/sangkan-dev) · 2026

</div>
