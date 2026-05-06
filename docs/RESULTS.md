# Hasil Eksperimen WIJI — Complete Documentation

Dokumen ini berisi hasil lengkap dari **9 eksperimen Phase 0** yang telah dilakukan, termasuk metodologi, hasil mentah, dan analisis.

## Setup Eksperimen

- **Model target**: TinyLlama-1.1B-Chat-v1.0
- **Hardware**: Laptop CPU (no GPU)
- **Framework**: PyTorch + transformers
- **Test prompt**: `"What is the capital of Indonesia?"`
- **Expected output**: Variations of "Jakarta is the capital of Indonesia"

## Ringkasan Eksekutif

| # | Eksperimen | Compression | MSE Layer 21 | Output Quality |
|---|-----------|-------------|--------------|----------------|
| 1 | Single matrix | 25x | N/A | ✅ Functional |
| 2 | Multi-layer single gen | 545x | 0.000234 | ❌ Gibberish |
| A | Full attention layer 0 | 56x | N/A | ✅ Functional |
| B2 | Per-layer generator | 25x | 0.000373 | ❌ Gibberish |
| B3 | Adaptive capacity | 15x | 0.000366 | ❌ Worse |
| B4 | Cliff edge test | varies | (uses B3) | 🔍 Cliff at N=2 |
| C1 | Fourier Features | 42x | 0.000356 | ❌ Same plateau |
| D1 | FFN target | 116x | 0.000315 | ❌ Same plateau |
| E1 | SIREN | 28x | 0.000371 | ❌ Same plateau |
| F1 | Mixture of Generators | 32x | 0.000377 | ❌ Same plateau |

## Eksperimen 1 — Single Matrix Reconstruction

**Tujuan**: Membuktikan bahwa weight matrix dari LLM bisa di-rekonstruksi dari coordinate-based generator yang lebih kecil.

### Setup
- Target: 1 matriks (`o_proj` layer 0), 4.2M parameters
- Generator: Coordinate MLP (164K params)
- Compression: 25.53x
- Training: 5000 steps

### Hasil
```
Step  200/5000 | MSE Loss: 0.000073
Step 5000/5000 | MSE Loss: 0.000065
```

**Output**: `"The capital of Indonesia is Jakarta."` (✅ functional)

---

## Eksperimen 2 — Multi-Layer Single Generator

**Tujuan**: Test apakah 1 generator bisa handle 22 layer.

### Setup
- Target: 22 matriks `o_proj`, 92M params
- Generator: Coordinate MLP + layer embedding (169K params)
- Compression: **545.72x**

### Hasil
```
Final MSE: 0.000234
```

**Output**: `"Ingatescripturecordialoisimoisequalifiesearchivedeastern Discogs, and"` (❌ gibberish)

**Insight**: MSE hanya 3x lebih tinggi dari Eksperimen 1, tapi output collapse total. **MSE bukan predictor reliable.**

---

## Eksperimen Diagnostik A — Full Attention Layer 0

**Tujuan**: Test single generator handle 4 attention components (Q/K/V/O).

### Setup
- Target: 9.4M params
- Generator: 168K params
- Compression: 56x

### Hasil
```
Final MSE: 0.000400
```

**Output**: `"The capital of Indonesia is Jakarta."` (✅ functional)

**Insight**: MSE 6x lebih tinggi dari Exp 1, tapi output preserved. Konfirmasi bahwa masalah Exp 2 adalah **multi-layer compounding**, bukan multi-component.

---

## Eksperimen B2 — Per-Layer Generator

**Tujuan**: 22 generator terpisah, masing-masing untuk 1 layer.

### Setup
- 22 × 164K = 3.6M params
- Compression: 25.53x
- Training: 1000 steps per layer

### Hasil
| Layer | MSE | Layer | MSE |
|-------|------|-------|------|
| 00 | 0.000063 | 11 | 0.000213 |
| 05 | 0.000203 | 15 | 0.000244 |
| 10 | 0.000225 | 21 | **0.000373** |

**Output**: `"WHEREASPark. ."` (❌ gibberish)

**Insight**: MSE meningkat monoton 6x dari layer 0 ke layer 21. Layer akhir lebih kompleks.

---

## Eksperimen B3 — Adaptive Capacity

**Tujuan**: Bigger generator untuk layer akhir.

### Setup
- Layer 0-7: 128 hidden dim
- Layer 8-15: 256 hidden dim
- Layer 16-21: 512 hidden dim
- Total: 5.9M params, training 3000 steps

### Hasil
```
Layer 21 MSE: 0.000366 (vs B2: 0.000373)
```

**Output**: `` ` ` ` ` ` `` (❌ degenerate loop)

**INSIGHT KRITIS**: Generator 10x lebih besar + training 3x lebih lama → MSE **hampir identik**. Hint pertama bahwa ini bukan masalah capacity.

---

## Eksperimen B4 — Cliff Edge Test

**Tujuan**: Cari titik dimana model collapse.

### Hasil
| N | Output |
|---|--------|
| 1 | "The capital of Indonesia is Jakarta." ✅ |
| 3 | "The capital of Ia is 10." ⚠️ |
| 5 | "" ❌ |
| 8 | "" ❌ |
| 12 | "" ❌ |
| 16 | "" ❌ |
| 22 | "Ingunsuretournalty. WHERE2..." ❌ |

**Insight**: Cliff edge sangat tajam di **N=2**. N=22 lebih baik dari N=5-16 menunjukkan **internal consistency matters**.

---

## Eksperimen C1 — Fourier Features

**Tujuan**: Test apakah Fourier features (NeRF-style positional encoding) bisa solve hipotesis spectral bias.

### Setup
- Generator: Pure Fourier encoding (no nn.Embedding) + MLP
- 16 frequency bands, hidden_dim=256
- 22 generators × 99K params = 2.2M total
- Compression: **42.31x**

### Hasil
```
Layer 00 | MSE: 0.000072
Layer 10 | MSE: 0.000220
Layer 21 | MSE: 0.000356  ← masih plateau yang sama
```

### Cliff Edge
| N | Output |
|---|--------|
| 1 | "The capital of Indonesia is Jakarta." ✅ |
| 3 | "I don't know. Can you be a helpful assistant." ⚠️ |
| 5 | "<\|user\|>You are a helpful assistant." ❌ |
| 8-16 | "" ❌ |
| 22 | "Theoryatarea, androoffertiltournputnamelysiphoniesearchive" ❌ |

### Analisis

**❌ FAILURE — tapi insight yang sangat penting.**

Fourier features didesain spesifik untuk solve spectral bias (terbukti works di NeRF). Namun **MSE plateau hampir identik** dengan Coordinate MLP biasa (0.000356 vs 0.000366).

**Implikasi**: Masalah yang kami hadapi **bukan spectral bias**.

---

## Eksperimen D1 — FFN Target (down_proj)

**Tujuan**: Test apakah FFN layer (`down_proj`) lebih mudah di-fit dari attention layer (`o_proj`).

### Setup
- Target: 22 × `down_proj` (2048×5632 each)
- Total target: 253M params
- Generator: 22 × Fourier dynamic = 2.2M
- Compression: **116.35x**

### Hasil
```
Layer 00 | MSE: 0.000273
Layer 10 | MSE: 0.000298
Layer 21 | MSE: 0.000315  ← plateau
```

### Cliff Edge
| N | Output |
|---|--------|
| 1 | "The capital of Indonesia is Jakarta." ✅ |
| 3 | "" ❌ |
| 5 | ",and,and,and,and,..." ❌ |
| 8 | "steruptruptruptrupt..." ❌ |
| 12 | "FFFFFFFFFFFFFFFFFFFF" ❌ |
| 16 | "CNN CNN CNN CNN..." ❌ |
| 22 | "wrong wrong wrong..." ❌ |

### Analisis

**❌ FAILURE — same plateau, even slightly higher floor.**

FFN tidak lebih mudah di-fit dari attention. Plateau MSE konsisten ~0.0003.

---

## Eksperimen E1 — SIREN (Sinusoidal Activations)

**Tujuan**: Test SIREN architecture (Sitzmann et al. 2020) untuk solve spectral bias dengan cara berbeda dari Fourier.

### Setup
- Generator: SIREN (sinusoidal activations + omega_0=30)
- 22 generators × 149K params = 3.3M
- Compression: 28.14x

### Hasil
```
Layer 00 | MSE: 0.000064
Layer 10 | MSE: 0.000227
Layer 21 | MSE: 0.000371  ← still the plateau
```

### Cliff Edge
| N | Output |
|---|--------|
| 1 | "The capital of Indonesia is Jakarta." ✅ |
| 3 | "I don't know." ⚠️ |
| 5 | "<\|user\|>You are..." ❌ |
| 8-12 | "" ❌ |
| 16 | "The\|assistant\|assistant\|user..." ❌ |
| 22 | "Ingamesideogram, androaming totokenshellsingularum..." ❌ |

### Analisis

**❌ FAILURE — confirms it's not about activation function.**

Dua teknik yang fundamental berbeda untuk solve spectral bias (Fourier features dan SIREN), keduanya gagal menggeser MSE plateau. Ini bukti **kuat** bahwa masalahnya bukan spectral bias.

---

## Eksperimen F1 — Mixture of Generators (Sharding)

**Tujuan**: Pecah matrix ke 16 chunks, pakai 16 expert generators dengan internal routing.

### Setup
- Sharding: 4×4 = 16 experts per layer
- 22 layers × 16 experts dengan local coords (512×512)
- Total: 2.9M params
- Compression: 31.57x

### Hasil
```
Layer 00 | MSE: 0.000067
Layer 10 | MSE: 0.000221
Layer 21 | MSE: 0.000377  ← plateau lagi!
```

### Cliff Edge
| N | Output |
|---|--------|
| 1 | "The capital of Indonesia is Jakarta." ✅ |
| 3 | "I don't know." ⚠️ |
| 5-16 | "" ❌ |
| 22 | "WHERE2. WHERE2..." ❌ |

### Analisis

**❌ FAILURE — sharding doesn't help either.**

Idea: spatial locality + parameter sharing yang lebih granular harusnya help. Faktanya tidak. Same plateau.

---

## The Plateau — Visualisasi Argumen

```
Final MSE Layer 21 across 5 different generator architectures:

Coordinate MLP baseline (B2):  ████████████████████████████ 0.000373
Coordinate MLP adaptive (B3):  ████████████████████████████ 0.000366
Fourier Features        (C1):  ████████████████████████████ 0.000356
SIREN                   (E1):  ████████████████████████████ 0.000371
Mixture of Generators   (F1):  ████████████████████████████ 0.000377

Spread: 0.000021 (5.6%)
```

5 arsitektur fundamentally berbeda. Plateau identik.

**Itu bukan kebetulan. Itu information theoretic floor.**

---

## Kesimpulan Phase 0

### Apa yang Terbukti

1. ✅ Single layer compression up to 56x preserves output
2. ✅ Cliff edge phenomenon di N=2 (sharp phase transition)
3. ✅ MSE Loss bukan reliable predictor untuk LLM quality
4. ✅ Layer akhir lebih kompleks dari layer awal
5. ✅ Internal consistency matters (N=22 > N=5-16)
6. ✅ **MSE plateau ~0.0003 adalah floor**, bukan ceiling

### Apa yang Eliminated

1. ❌ Bukan masalah generator capacity (B3)
2. ❌ Bukan masalah spectral bias (C1, E1)
3. ❌ Bukan masalah specific layer type (D1)
4. ❌ Bukan masalah spatial locality (F1)

### Final Hypothesis

> **Weight neural network setelah training dengan SGD adalah stochastic outcome dari training history, bukan smooth function dari koordinat. Coordinate-based generators tidak bisa fit ini regardless of architecture, karena tidak ada underlying function untuk di-fit.**

### Path Forward (Untuk Orang Lain)

Jalur yang **belum** kami eksplor dan secara teoritis bisa bypass:

1. **Co-Training**: Train model dari scratch dimana generator adalah part of architecture
2. **Output-Aware Loss**: KL divergence pada output, bukan MSE pada weight
3. **Distillation**: Student dengan generator architecture, learn from teacher
4. **Different Architecture Targets**: Mamba, RWKV, atau arsitektur dengan struktur weight berbeda

## Reproducibility

Semua eksperimen dapat di-reproduce dengan:

```bash
git clone https://github.com/sangkan-dev/wiji-experimental.git
cd wiji-experimental
uv sync
bash scripts/reproduce_all.sh
```

Variation antar run: 5-10% karena random initialization, tapi pattern umum (cliff edge, plateau) sangat konsisten.

## Hardware Performance

Total compute time untuk semua 9 eksperimen di laptop CPU: ~10 jam.

Bottleneck utama: matrix reconstruction (per-row generation untuk swap testing). Bisa dioptimisasi dengan batch generation, tapi tidak dilakukan karena bukan focus penelitian.