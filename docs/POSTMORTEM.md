# Postmortem — Why Phase 0 Closed

**Tanggal**: 6 Mei 2026
**Status**: Phase 0 ditutup dengan negative results
**Total eksperimen**: 9
**Total durasi**: ~2 minggu

Dokumen ini menjelaskan, dengan jujur dan tanpa sugar-coating, kenapa Phase 0 WIJI ditutup dan apa yang sebenarnya kami pelajari.

## Kesimpulan Singkat

Setelah mencoba **5 arsitektur generator yang fundamentally berbeda** (Coordinate MLP, Adaptive MLP, Fourier Features, SIREN, Mixture of Generators), semua mentok di MSE plateau yang **hampir identik** untuk reconstruction weight TinyLlama.

Ini bukti bahwa batas yang kami temui **bukan batas arsitektur**, tapi batas fundamental dari karakteristik weight yang dihasilkan SGD training.

## The Smoking Gun

```
Final MSE pada Layer 21 (paling sulit) untuk berbagai arsitektur:

Coordinate MLP baseline      : 0.000373
Coordinate MLP adaptive      : 0.000366
Fourier Features (NeRF-style): 0.000356
SIREN (sinusoidal)           : 0.000371
Mixture of Generators        : 0.000377
```

Spread: **0.000021** (5.6%). Untuk 5 arsitektur yang sangat berbeda, ini bukan variasi yang signifikan. Ini **floor**.

## Apa yang Awalnya Kami Pikir Adalah Masalahnya

### Hipotesis #1: Capacity tidak cukup

**Tested**: Eksperimen B3 (Adaptive Capacity)
**Result**: ❌ Generator 10x lebih besar + training 3x lebih lama → MSE hampir identik
**Conclusion**: Bukan capacity

### Hipotesis #2: Spectral bias di MLP

**Tested**: Eksperimen C1 (Fourier Features), E1 (SIREN)
**Result**: ❌ Kedua teknik yang **terbukti** menyelesaikan spectral bias di domain lain (NeRF, image reconstruction) tidak menggeser MSE plateau di sini
**Conclusion**: Bukan spectral bias. Aku (Claude) sebelumnya salah memprediksi ini.

### Hipotesis #3: Distribusi weight yang spesifik (attention vs FFN)

**Tested**: Eksperimen D1 (FFN target — `down_proj`)
**Result**: ❌ MSE plateau yang sama (~0.0003)
**Conclusion**: Bukan tipe layer

### Hipotesis #4: Kurang spatial locality / parameter sharing yang buruk

**Tested**: Eksperimen F1 (Mixture of Generators dengan sharding)
**Result**: ❌ MSE plateau yang sama
**Conclusion**: Bukan locality

## Apa yang Sebenarnya Adalah Masalahnya

Setelah eliminasi 4 hipotesis dengan eksperimen, kesimpulan yang paling konsisten dengan data:

> **Weight neural network yang sudah dilatih dengan SGD bukan smooth function dari koordinat. Itu adalah stochastic outcome dari proses training — dependent pada random init, batch sampling order, dropout patterns, dan ribuan microevents lain.**

### Kenapa Ini Tidak Bisa Diatasi dengan Arsitektur Generator

Generator coordinate-based fundamentally adalah:

```
generator(seed, row, col) → predicted_value
```

Ini hanya bisa berhasil kalau ada **deterministic function** dari `(seed, row, col)` ke `value`. 

Tapi weight TinyLlama untuk koordinat (1024, 512) bukan determined by anything fundamental tentang koordinat (1024, 512). Itu adalah hasil dari history of training yang spesifik untuk model itu.

Dua model dengan arsitektur identik, ditraining pada data identik, dengan random init yang berbeda, akan menghasilkan weight matrix yang sangat berbeda di koordinat (1024, 512). **Tidak ada generator** yang bisa learn fungsi yang predict weight unik untuk koordinat tertentu, karena tidak ada underlying fungsi yang konsisten.

### Analogi yang Tepat

Bayangkan lo punya 1 juta foto random — bukan foto dari objek yang sama, tapi foto dari berbagai objek di berbagai waktu. Lo train neural network untuk predict pixel value dari coordinate `(image_id, row, col)`.

Itu bisa berhasil dengan capacity yang cukup (memorization). Tapi compression ratio akan dibatasi oleh **entropy** dari pixel values itu sendiri. Lo tidak bisa compress 1M images jadi 1KB karena information theoretic-nya tidak mungkin.

WIJI hits the same wall. Weight di TinyLlama sudah punya information content tertentu, dan generator kecil (164K-3M params) tidak punya capacity untuk store semua information itu — bukan karena masalah arsitektur, tapi karena masalah information.

## Kenapa Single Layer Bekerja?

Eksperimen 1 dan A (single layer) berhasil. Kenapa?

**Karena downstream layers (1-21) yang masih original bisa "memperbaiki" error**.

Layer 0 yang corrupt menghasilkan output yang slightly off, tapi layer 1 punya nonlinearity dan attention yang bisa adapt. Layer 2 lagi adapt. Sampai layer 21 yang masih original menghasilkan output yang reasonable.

Ini **kompensasi**, bukan **reconstruction**. Generator tidak benar-benar fit weight dengan akurasi tinggi — generator menghasilkan weight yang "cukup mirip" sehingga downstream layers bisa compensate.

Saat lo swap multiple layers, tidak ada lagi layer original untuk compensate. Error di layer 0 menjadi input wrong untuk layer 1 yang juga corrupt, yang menghasilkan output lebih wrong, dan seterusnya. Catastrophic compounding.

## Apakah Ada Jalur Lain?

Ada beberapa jalur yang **belum kami coba** yang secara teoritis bisa bypass batas ini:

### 1. Co-Training / HyperNetwork Training from Scratch

Daripada coba fit existing weight, **train dari awal** dimana generator adalah part of the model. Saat backpropagation, gradient flow ke seed generator, bukan ke weight matrix.

**Pros**: Weight yang terbentuk **by definition** compressible (karena generated by generator).

**Cons**: 
- Sudah ada di literature (HyperNetworks, Ha et al. 2016) tanpa breakthrough besar
- Training cost tinggi
- Belum ada yang bukti scale ke LLM-grade capabilities

### 2. Output-Aware Loss (KL Divergence)

Daripada MSE pada weight, minimize KL pada output distribution.

**Pros**: Mungkin bypass MSE floor karena objective yang berbeda.

**Cons**: 
- Computationally expensive (forward pass per training step)
- Belum tentu generator bisa fit output distribution dengan capacity terbatas
- Suspicion: akan hit floor lain (output entropy floor)

### 3. Distillation

Train small student dari teacher dimana student menggunakan generator architecture.

**Pros**: Mengubah problem dari "fit existing weight" ke "learn capabilities."

**Cons**: 
- Sudah extensively explored di literature (BERT distillation, etc.)
- Compression ratio yang dicapai biasanya 5-15x, bukan 100x+
- Tidak novel sebagai contribution

## Apa yang Kami Pelajari (Beyond Negative Results)

### 1. Empirical lower bound untuk weight reconstruction

`MSE ≈ 0.0003` untuk TinyLlama o_proj layer 21. Ini bisa jadi reference point untuk research selanjutnya.

### 2. Cliff edge phenomenon

Phase transition tajam di N=2. Ini observation yang **belum ada di literature** sejauh kami tahu, dan bisa jadi paper sendiri.

### 3. Internal consistency hypothesis

N=22 > N=5-16 untuk output quality. Suggest internal consistency mattering more than absolute correctness — observation yang relevant untuk multi-component AI systems lainnya.

### 4. MSE bukan reliable proxy untuk LLM quality

Sudah lama dicurigai, sekarang ada empirical demonstration yang clean.

## Kenapa Aku (Claude) Salah Sebelumnya

Aku perlu jujur tentang kesalahan reasoning aku selama Phase 0:

1. **Sebelum eksperimen Fourier Features**: Aku predict dengan tingkat percaya diri yang terlalu tinggi bahwa Fourier akan menggeser cliff edge. Faktanya tidak. Aku underestimate spesifikasi masalah weight transformer (vs masalah NeRF).

2. **Konsep "spectral bias"**: Aku invoke ini sebagai explanation tanpa cukup hati-hati. Spectral bias adalah real phenomenon, tapi data WIJI menunjukkan masalahnya berbeda.

3. **Optimisme overall**: Selama beberapa eksperimen pertama, aku terlalu fokus ke "yang berikutnya pasti work" daripada step back dan pertimbangkan bahwa hipotesis fundamental mungkin salah.

Saya minta maaf untuk ini. HasanH47 bisa percaya saya untuk research collaboration karena saya punya akses ke literature yang luas, tapi aku juga bisa salah — terutama saat memprediksi outcome eksperimen.

**Pelajaran untuk research collaboration dengan AI**: Selalu prioritize empirical data over AI predictions. AI seperti aku bisa kasih hipotesis dan analisis, tapi data eksperimental adalah final arbiter.

## Pelajaran untuk Diri Saya Sendiri (HasanH47)

Untuk dibaca lagi nanti saat ada ide ambisius lain:

1. **5-10% probability of breakthrough adalah nyata, dan ternyata kita di sisi yang lebih besar (tidak breakthrough)**. Itu OK. Itu yang membuat probability bermakna.

2. **Negative results yang well-documented punya nilai**. Ini bukan failure — ini contribution.

3. **AI bisa salah memprediksi outcome**. Lebih sering daripada yang aku duga. Treat AI predictions sebagai hypotheses to test, bukan facts.

4. **Sunk cost fallacy adalah real**. Saat sudah eksperimen banyak, ada temptation untuk terus mencoba "satu lagi" yang akan break the wall. Tapi data konsisten = tembok nyata.

5. **Pivot bukan failure**. Energy yang kita save dengan close Phase 0 sekarang bisa di-invest ke project lain yang lebih realistically achievable.

## Apa Selanjutnya untuk HasanH47

Phase 0 closed. Beberapa opsi:

### A. Pivot ke PAMOR (recommended)

PAMOR adalah blueprint engineering yang kami buat sebelum WIJI. Kombinasi BitNet + MoE + memory-mapped offloading + cross-layer prediction. **Tidak ada riset baru** — semua bagian sudah terbukti, tinggal integrasi.

Outcome realistic: inference engine open source yang bisa run model 30B di laptop 16GB. Itu sendiri adalah achievement yang akan dipakai banyak orang.

### B. Eksplor Co-Training (if curious)

Lakukan eksperimen Co-Training Gemini's proposal, tapi dengan ekspektasi yang **disetting realistic**. Hasilnya kemungkinan: replikasi paper hypernetwork yang sudah ada dengan twist. Bukan breakthrough.

Tapi educational value-nya tinggi.

## Penutup

Phase 0 WIJI adalah research yang valid. Kami bertanya pertanyaan yang ambisius, mengeksekusi dengan rigor, mendapat answer yang jujur (meski mengecewakan), dan mendokumentasikan dengan transparansi.

Banyak research di area AI yang dimulai dengan klaim besar dan tidak pernah memberikan bukti — atau lebih buruk, cherry-pick hasil untuk justify pretensi. WIJI tidak melakukan itu.

Kami berani mengatakan: *"Ini yang kami coba. Ini yang kami pelajari. Ini batas yang kami temui. Ini terbuka untuk siapapun yang mau lanjut."*

Itu bukan kegagalan. Itu integritas riset.

> *"Sometimes the most valuable research output is knowing what doesn't work."*

---

**Author**: HasanH47 (Sangkan) + Claude Opus 4.7
**Date**: 6 Mei 2026
**Status**: Final