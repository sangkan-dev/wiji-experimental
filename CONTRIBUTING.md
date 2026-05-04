# Contributing to WIJI

Terima kasih atas ketertarikan Anda! WIJI adalah project riset terbuka, dan kontribusi sangat dihargai.

## Prinsip Kontribusi

WIJI adalah **research project**, bukan production code. Yang utama: **rigor ilmiah** dan **dokumentasi yang baik**.

## Cara Berkontribusi

### 1. Reproduce Eksperimen yang Ada

Cara paling mudah memulai. Run salah satu eksperimen di folder `experiments/`, dan laporkan hasil di [Issues](https://github.com/sangkan-dev/wiji-experimental/issues).

Format laporan:
```markdown
**Eksperimen**: exp01_single_matrix.py
**Hardware**: [CPU/GPU model, RAM]
**Hasil**: [output dan MSE]
**Catatan**: [any deviation dari expected]
```

### 2. Bikin Eksperimen Baru

Beberapa ide eksperimen yang dibutuhkan:

- **Fourier Features encoding** untuk solve spectral bias
- **Different model**: Coba di Qwen 1.5B, Llama 3.2 1B, Phi-3 mini
- **Different layer types**: Test di FFN layers (`gate_proj`, `up_proj`, `down_proj`)
- **Output-aware loss**: KL-divergence implementation
- **Streaming inference**: Implementasi yang load 1 layer at a time

Format eksperimen baru:
1. File: `experiments/expXX_descriptive_name.py`
2. Header: docstring dengan tujuan, hipotesis, hasil
3. Update `docs/RESULTS.md` dengan hasil
4. Update `docs/INSIGHTS.md` jika ada insight baru

### 3. Improve Documentation

- Translate dokumen ke bahasa lain
- Bikin tutorial video / blog post
- Improve clarity dari existing docs
- Add diagrams / visualisasi

### 4. Critique Methodology

Sangat penting. WIJI bisa salah di banyak tempat. Kritik konstruktif membantu rigor.

Buka issue dengan tag `methodology-critique` untuk:
- Flaws di experimental setup
- Statistical issues
- Missing controls
- Confounding variables

## Pull Request Guidelines

### Setup Development

```bash
git clone https://github.com/sangkan-dev/wiji-experimental.git
cd wiji-experimental
uv sync
```

### Commit Message Format

Gunakan conventional commits:

```
<type>(<scope>): <description>

[optional body]
```

Types:
- `exp:` Eksperimen baru
- `docs:` Documentation
- `fix:` Bug fix
- `feat:` Fitur baru di tooling
- `refactor:` Code restructuring tanpa change behavior
- `test:` Add/modify tests

Examples:
```
exp(fourier): add Fourier features encoding experiment
docs(insights): add observation about cliff edge behavior
fix(exp01): correct matrix dimensions for Qwen model
```

### Code Style

- Use Python 3.10+ syntax
- Type hints encouraged tapi tidak mandatory
- Docstrings untuk semua function eksperimen (tujuan, hipotesis, hasil)
- Comments untuk explain "why", bukan "what"
- Maximum line length: 100 chars

## Filosofi Kontribusi

### Berani gagal, dokumentasikan kegagalan

Kontribusi terbaik kadang adalah eksperimen yang gagal — selama didokumentasi dengan baik dan menjelaskan **kenapa** gagal.

### Question everything

Termasuk konsep dasar WIJI itu sendiri. Kalau lo merasa pendekatan ini fundamentally flawed, lo punya argumen yang valid, **buat issue dengan tag `philosophical-challenge`**. Kita welcome debate ilmiah.

### Indonesian first, but not exclusive

Project ini punya identity Indonesia (naming, philosophy), tapi kontribusi welcome dari siapapun. Bahasa Indonesia atau Inggris keduanya OK untuk komunikasi.

## Code of Conduct

Singkat:
- Be respectful
- Be intellectually honest
- Don't claim work that isn't yours
- Cite properly
- Help others learn

## Lisensi

Kontribusi otomatis di-license di bawah MIT, sama dengan project utama.

## Pertanyaan?

- Buka issue dengan tag `question`
- Email: [your-email@example.com]
- LinkedIn: [HasanH47](https://linkedin.com/in/hasanh47)

---

> *"The best way to predict the future is to invent it."* — Alan Kay
