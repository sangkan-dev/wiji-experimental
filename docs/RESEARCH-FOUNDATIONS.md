# Research Foundations

Dokumen ini berisi paper-paper riset yang menjadi fondasi konseptual dan teknis untuk WIJI. Bukan exhaustive bibliography, tapi paper yang **secara langsung relevan** dengan ide WIJI.

## 1. Implicit Neural Representations (INR) & Hypernetworks

### D'OH: Decoder-Only Random Hypernetworks for Implicit Neural Representations
- **Authors**: Gordon, C. et al.
- **Venue**: ACCV 2024
- **arXiv**: [2403.19163](https://arxiv.org/abs/2403.19163)

**Relevansi untuk WIJI**: Paper ini membuktikan bahwa weight INR bisa di-generate dari latent code kecil + random projection. Mereka pakai untuk image compression, tapi konsep yang sama bisa diaplikasikan ke LLM weight reconstruction.

**Quote kunci**: *"As only the latent code, biases, and an integer random seed need to be communicated to reconstruct signals."*

### Recursive Self-Similarity in Deep Weight Spaces
- **Authors**: Moharil et al.
- **arXiv**: [2503.14298](https://arxiv.org/abs/2503.14298)
- **Tahun**: Maret 2025

**Relevansi untuk WIJI**: Paper ini secara empiris membuktikan bahwa weight space neural network punya **dimensi fractal Hausdorff-Besicovitch yang terukur**. Artinya: weight transformer punya struktur self-similar yang bisa dieksploitasi untuk kompresi.

**Implikasi**: Memberikan justifikasi matematis bahwa fractal hypernetwork (yang kita pertimbangkan untuk Phase 1) bukan halusinasi.

## 2. Quantization & Efficiency

### BitNet b1.58 — The Era of 1-bit LLMs
- **Authors**: Ma, S. et al.
- **arXiv**: [2402.17764](https://arxiv.org/abs/2402.17764)
- **Tahun**: 2024

**Relevansi untuk WIJI**: Membuktikan bahwa weight ternary {-1, 0, +1} bisa achieve performance comparable dengan FP16. Ini suggest bahwa weight transformer punya redundansi yang bisa di-compress drastis tanpa kehilangan kapabilitas.

**Quote kunci**: *"BitNet b1.58 matches the perplexity and end-task performance of full-precision baseline."*

### BitDistill: Distilling Full-Precision LLMs to 1.58-bit
- **arXiv**: [2510.13998](https://arxiv.org/abs/2510.13998)
- **Tahun**: Oktober 2025

**Relevansi untuk WIJI**: Distillation pipeline FP16 → 1.58-bit. Suggest path untuk WIJI: ambil model existing, distill ke representasi WIJI tanpa retrain dari scratch.

## 3. Mixture of Experts & Sparse Activation

### Pre-Attention Expert Prediction and Prefetching for MoE LLMs
- **Authors**: Zhu, S. et al.
- **arXiv**: [2511.10676](https://arxiv.org/abs/2511.10676)
- **Tahun**: November 2025

**Relevansi untuk WIJI**: Achieve 93-97% accuracy untuk prediksi expert MoE pre-attention. Konsep yang serupa (predict mana weight yang akan dibutuhkan) relevan untuk WIJI streaming inference.

### PowerInfer / PowerInfer-2
- **Author**: Song, Y. et al.
- **Venue**: SOSP 2024 / mobile inference 2025

**Relevansi untuk WIJI**: Demonstrate 11.6x speedup untuk LLM inference dengan activation prediction + smart caching. Pattern yang mirip untuk WIJI: predict mana tile weight yang akan digunakan.

## 4. Alternative Architectures

### Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- **Authors**: Gu, A., Dao, T.
- **arXiv**: [2312.00752](https://arxiv.org/abs/2312.00752)

**Relevansi untuk WIJI**: Alternatif transformer yang lebih efficient. Kalau WIJI scale ke production, mungkin lebih mudah di-implementasikan di SSM daripada transformer (karena KV cache yang berbeda).

### Fractal Generative Models
- **arXiv**: [2502.17437](https://arxiv.org/abs/2502.17437)
- **Tahun**: Februari 2025

**Relevansi untuk WIJI**: Self-similar autoregressive blocks. Konsep fractal di generation domain. Bisa diadaptasi untuk weight generation di WIJI Phase 1.

### Titans: Learning to Memorize at Test Time
- **Author**: Behrouz, A. et al.
- **arXiv**: [2501.00663](https://arxiv.org/abs/2501.00663)
- **Tahun**: 2025

**Relevansi untuk WIJI**: Neural long-term memory yang belajar di test time. Konsep "weight yang berubah dinamis" sejalan dengan visi WIJI tentang weight sebagai computation.

## 5. Hardware & Computation Paradigms

### Analog In-Memory Computing for Neural Networks
- **Source**: IBM Research, 2025
- **Link**: https://research.ibm.com/blog/how-can-analog-in-memory-computing-power-transformer-models

**Relevansi untuk WIJI**: Hardware bergerak ke arah memory + compute jadi satu. WIJI software-side sangat kompatibel dengan paradigma ini — kalau WIJI berhasil, mapping ke analog hardware menjadi natural.

### Hafnium Oxide Neuromorphic Chip
- **Source**: University of Cambridge, April 2026
- **Coverage**: ScienceDaily

**Relevansi untuk WIJI**: Chip yang menyimpan dan memproses informasi di tempat yang sama, 70% energy reduction. Validasi lebih lanjut bahwa von Neumann bottleneck bisa di-bypass.

## 6. Spectral Bias & Coordinate Networks

### NeRF: Representing Scenes as Neural Radiance Fields
- **Authors**: Mildenhall, B. et al.
- **Venue**: ECCV 2020

**Relevansi untuk WIJI**: Pioneer dari coordinate-based MLP dengan Fourier features untuk fit high-frequency functions. Pattern yang langsung applicable untuk WIJI generator.

**Pelajaran**: NeRF awalnya gagal fit detail tinggi — sampai mereka tambah positional encoding (Fourier features). Ini direct precedent untuk masalah spectral bias kita di B3.

### SIREN: Implicit Neural Representations with Periodic Activation Functions
- **Authors**: Sitzmann, V. et al.
- **Venue**: NeurIPS 2020

**Relevansi untuk WIJI**: Sinusoidal activation functions (instead of ReLU/GELU) untuk fit high-frequency signals. Alternatif dari Fourier features untuk solve spectral bias.

### Fourier Features Let Networks Learn High Frequency Functions
- **Authors**: Tancik, M. et al.
- **Venue**: NeurIPS 2020

**Relevansi untuk WIJI**: Theoretical foundation kenapa Fourier encoding works untuk spectral bias. Bacaan wajib sebelum implementasi Phase 1.

## 7. Lottery Ticket Hypothesis & Network Pruning

### The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
- **Authors**: Frankle, J., Carbin, M.
- **Venue**: ICLR 2019

**Relevansi untuk WIJI**: Suggest bahwa neural network punya sparse subnetwork yang sufficient untuk task. Memperkuat hipotesis bahwa weight punya redundansi yang besar — kompresi WIJI punya basis teoritis.

## 8. Locality-Sensitive Hashing

### LSH untuk Approximate Nearest Neighbor
- **Classic reference**: Indyk, P., Motwani, R. (1998)

**Relevansi untuk WIJI**: Untuk speculative skip via semantic hashing (Phase 1 inovasi #2). Menggunakan LSH pada input embedding untuk identify computation yang sudah pernah dilakukan.

## 9. Continual Learning & Catastrophic Forgetting

### Continual Learning in Neural Networks
- Multiple papers, ongoing research area

**Relevansi untuk WIJI**: Adaptation layer di WIJI (delta seed) berhubungan langsung dengan continual learning. Kalau delta seed kecil bisa adapt model ke domain baru tanpa forgetting, ini contribution besar untuk continual learning literature.

## 10. Background Reading untuk Pendatang Baru

Kalau lo baru di area ini, urutan baca yang aku rekomendasikan:

1. **Mulai dari NeRF paper** — paling intuitive, dengan visualisasi yang bagus
2. **Lanjut ke SIREN** — alternatif elegant
3. **Baca BitNet b1.58** — untuk understand quantization extreme
4. **Baca D'OH paper** — untuk understand hypernetwork untuk compression
5. **Baca paper Recursive Self-Similarity (Mar 2025)** — untuk fondasi matematis fractal di weight space
6. **Baru baca PRD WIJI** dengan konteks lengkap

## 11. Ongoing Research yang Perlu Dipantau

Area-area yang aktif berkembang dan akan affect WIJI:

- **Mixture of Experts research** — terutama efficient routing
- **Implicit neural representations** — generalisasi ke modality lain
- **Neuromorphic computing** — hardware breakthrough
- **State space models** — alternatif transformer
- **Test-time computation scaling** — paradigma baru yang sejalan dengan WIJI

## Pesan Penutup

Membaca paper bukan tujuan, **memahami insight** adalah tujuan. Setiap paper di list ini ada karena ada satu insight kunci yang relevan untuk WIJI.

Kalau lo baca dan tidak nemu insight kuncinya, baca lagi. Kalau masih tidak nemu, kemungkinan paper itu memang tidak relevan untuk approach lo — itu juga insight valid.

> *"The most important thing in research is to know what you're trying to figure out."* — paraphrased from Hamming
