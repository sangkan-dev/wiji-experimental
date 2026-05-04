## Experiment Results Log

This file contains the actual outputs from running each experiment, copied from terminal logs.

---

## Experiment 1: Single Matrix

```
=== EKSTRAKSI BERHASIL ===
Target Matriks  : Layer 0 -> o_proj
Bentuk (Shape)  : torch.Size([2048, 2048]) (Baris x Kolom)
Total Angka     : 4,194,304 parameter
Contoh 5 Angka  : [0.0019989013671875, -0.001068115234375, 0.00165557861328125, ...]
==========================

Target Weight  : 4,194,304 parameters (~16.7 MB)
Wiji Generator : 164,289 parameters (~0.63 MB)
Rasio Kompresi : 25.53x lebih kecil!

Step  200/5000 | Error (MSE Loss): 0.000073
Step 1000/5000 | Error (MSE Loss): 0.000073
Step 2000/5000 | Error (MSE Loss): 0.000074
Step 3000/5000 | Error (MSE Loss): 0.000067
Step 4000/5000 | Error (MSE Loss): 0.000074
Step 5000/5000 | Error (MSE Loss): 0.000065

=== TES INFERENCE ASLI (SEBELUM BRAIN SWAP) ===
Indonesia's capital is Jakarta.

=== TES INFERENCE TIRUAN (SETELAH BRAIN SWAP) ===
The capital of Indonesia is Jakarta.

STATUS: ✅ SUCCESS
```

---

## Experiment 2: Multi-Layer Single Generator

```
Target Weight (22 Layer) : 92,274,688 params (~367.0 MB)
Wiji Generator           : 169,089 params (~0.7 MB)
RASIO KOMPRESI GILA      : 545.72x lebih kecil!!!

Training Step  500/3000 | MSE Loss: 0.000251
Training Step 1000/3000 | MSE Loss: 0.000238
Training Step 1500/3000 | MSE Loss: 0.000233
Training Step 2000/3000 | MSE Loss: 0.000240
Training Step 2500/3000 | MSE Loss: 0.000234
Training Step 3000/3000 | MSE Loss: 0.000234

=== TES INFERENCE TIRUAN (SETELAH BRAIN SWAP 22 LAYER) ===
Ingatescripturecordialoisimoisequalifiesearchivedeastern Discogs, and

STATUS: ❌ FAILURE — Complete semantic collapse
INSIGHT: MSE only 3x higher than Exp 1, but output collapsed
```

---

## Experiment Diagnostik A: Full Attention Layer 0

```
Target Weight (4 Matriks) : 9,437,184 params (~36.0 MB)
Wiji Generator            : 168,513 params (~0.64 MB)
Rasio Kompresi            : 56.00x lebih kecil!

Training Step  500/3000 | MSE Loss: 0.000379
Training Step 1000/3000 | MSE Loss: 0.000305
Training Step 1500/3000 | MSE Loss: 0.000354
Training Step 2000/3000 | MSE Loss: 0.000334
Training Step 2500/3000 | MSE Loss: 0.000344
Training Step 3000/3000 | MSE Loss: 0.000407

=== TES INFERENCE TIRUAN (SETELAH BRAIN SWAP LAYER 0 FULL) ===
The capital of Indonesia is Jakarta.

STATUS: ✅ SUCCESS — Full attention layer compressed 56x
INSIGHT: MSE 6x higher than Exp 1 but output preserved
```

---

## Experiment B2: Per-Layer Generator

```
Target Weight (22 Matriks Asli) : 92,274,688 params (~352.0 MB)
Cluster 22 Wiji Generator       : 3,614,358 params (~13.79 MB)
Rasio Kompresi Horizontal       : 25.53x lebih kecil!

Layer 00 | Final MSE: 0.000063
Layer 01 | Final MSE: 0.000180
Layer 02 | Final MSE: 0.000195
Layer 03 | Final MSE: 0.000199
Layer 04 | Final MSE: 0.000198
Layer 05 | Final MSE: 0.000203
Layer 06 | Final MSE: 0.000198
Layer 07 | Final MSE: 0.000218
Layer 08 | Final MSE: 0.000216
Layer 09 | Final MSE: 0.000224
Layer 10 | Final MSE: 0.000225
Layer 11 | Final MSE: 0.000213
Layer 12 | Final MSE: 0.000245
Layer 13 | Final MSE: 0.000225
Layer 14 | Final MSE: 0.000241
Layer 15 | Final MSE: 0.000244
Layer 16 | Final MSE: 0.000252
Layer 17 | Final MSE: 0.000293
Layer 18 | Final MSE: 0.000305
Layer 19 | Final MSE: 0.000335
Layer 20 | Final MSE: 0.000339
Layer 21 | Final MSE: 0.000373

=== TES INFERENCE TIRUAN ===
WHEREASPark. .

STATUS: ❌ FAILURE
INSIGHT: MSE increases monotonically with layer depth (0.000063 → 0.000373, 6x)
```

---

## Experiment B3: Adaptive Capacity

```
Target Weight (22 Asli)     : 92,274,688 params (~352.0 MB)
Cluster 22 Auto-Scaled Wiji : 5,918,102 params (~22.58 MB)
Rasio Kompresi Akhir        : 15.59x lebih kecil!

Capacity:
- Layer 0-7:   hidden_dim=128 (small)
- Layer 8-15:  hidden_dim=256 (medium)
- Layer 16-21: hidden_dim=512 (large)

Training: 3000 steps per layer (3x more than B2)

Layer 00 | Final MSE: 0.000075
Layer 01 | Final MSE: 0.000202
...
Layer 21 | Final MSE: 0.000366

=== TES INFERENCE ===
``` ``` ``` ``` (degenerate loop)

STATUS: ❌ FAILURE
INSIGHT (Most important): MSE plateau is INDEPENDENT of capacity
- B2 Layer 21: 0.000373 (small generator, 1000 steps)
- B3 Layer 21: 0.000366 (10x bigger generator, 3000 steps)
- Difference: 2% — essentially identical

This proves SPECTRAL BIAS: coordinate MLPs hit hard limit.
```

---

## Experiment B4: Cliff Edge Test

```
Test points: N = 1, 3, 5, 8, 12, 16, 22 layers swapped

Test N=01 | Output: 'The capital of Indonesia is Jakarta.'
Test N=03 | Output: 'The capital of Ia is 10.'
Test N=05 | Output: ''
Test N=08 | Output: ''
Test N=12 | Output: ''
Test N=16 | Output: ''
Test N=22 | Output: 'Ingunsuretournalty. WHERE2. WHERE2. WHERE2.'

STATUS: 🔍 Cliff edge identified between N=1 and N=3

KEY INSIGHTS:
1. Cliff edge is SHARP, not gradual (phase transition behavior)
2. N=22 produces output while N=5-16 are empty
3. Internal consistency matters: same-distribution corruption tolerable,
   mixed-distribution corruption causes probability collapse
```

---

## Comparative Summary

| Exp | Compression | Final MSE | Output Quality |
|-----|-------------|-----------|----------------|
| 1 (1 matrix) | 25x | 0.000067 | ✅ Functional |
| A (full attn L0) | 56x | 0.000400 | ✅ Functional |
| 2 (22 layers, single gen) | 545x | 0.000234 | ❌ Gibberish |
| B2 (22 layers, per-layer) | 25x | 0.000063-0.000373 | ❌ Gibberish |
| B3 (adaptive capacity) | 15x | 0.000075-0.000366 | ❌ Worse |

**Conclusion**: Single-layer compression fundamentally works. Multi-layer compression hits a hard wall that capacity cannot solve.
