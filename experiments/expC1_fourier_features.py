"""
Eksperimen C1 — Fourier Features (The Equalizer)

Tujuan: Memecahkan masalah "Spectral Bias" pada Linear Network biasa
dengan menyuntikkan Fourier Features (Positional Encoding frekuensi tinggi).
Kita ingin melihat apakah ini bisa menggeser The Cliff Edge (Jurang Kematian)
melewati N=3.

Hipotesis: Fourier Features memungkinkan MLP kecil untuk me-render
distribusi weight layer akhir yang memiliki entropi (high-frequency) tinggi.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

class WijiGeneratorFourier(nn.Module):
    """
    Generator tanpa nn.Embedding. 
    Murni mengandalkan Fourier Transform dari koordinat baris/kolom.
    """
    def __init__(self, matrix_size=2048, seed_dim=64, num_freq=16, hidden_dim=256):
        super().__init__()
        self.matrix_size = matrix_size
        self.num_freq = num_freq
        self.seed = nn.Parameter(torch.randn(1, seed_dim))
        
        # Input: (Baris Freq x 2) + (Kolom Freq x 2) + Seed
        input_dim = (num_freq * 2 * 2) + seed_dim
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, row_indices, col_indices):
        # 1. Normalisasi koordinat menjadi skala [-1.0, 1.0]
        r_norm = (row_indices.float() / self.matrix_size) * 2.0 - 1.0
        c_norm = (col_indices.float() / self.matrix_size) * 2.0 - 1.0
        
        r_norm = r_norm.unsqueeze(-1)  # Shape: (batch, 1)
        c_norm = c_norm.unsqueeze(-1)  # Shape: (batch, 1)

        # 2. Bikin deret frekuensi (1, 2, 4, 8, 16...)
        freqs = (2 ** torch.arange(self.num_freq, device=r_norm.device)).float()
        
        # 3. Fourier Transform (Kalikan dengan frekuensi dan Pi)
        r_freq = r_norm * freqs * math.pi
        c_freq = c_norm * freqs * math.pi
        
        # 4. Ekstrak gelombang Sinus dan Cosinus
        r_fourier = torch.cat([r_freq.sin(), r_freq.cos()], dim=-1)
        c_fourier = torch.cat([c_freq.sin(), c_freq.cos()], dim=-1)
        
        # 5. Gabungkan dengan DNA (Seed)
        s = self.seed.expand(r_norm.shape[0], -1)
        x = torch.cat([r_fourier, c_fourier, s], dim=-1)
        
        return self.generator(x).squeeze(-1)


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("WIJI Eksperimen C1: Fourier Features & The Cliff Edge")
    print("=" * 60)
    
    print("\n[1/4] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.float32
    )

    prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is the capital of Indonesia?\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n[2/4] Mengekstrak 22 Matriks Asli & Inisialisasi Fourier Generators...")
    num_layers = 22
    original_weights = [model.model.layers[i].self_attn.o_proj.weight.data.clone() for i in range(num_layers)]
    
    # Bikin 22 Generator Fourier (Ukurannya sama semua, tidak perlu auto-scaling dulu)
    generators = nn.ModuleList([WijiGeneratorFourier() for _ in range(num_layers)])

    total_target_params = sum(t.numel() for t in original_weights)
    total_gen_params = sum(sum(p.numel() for p in g.parameters()) for g in generators)
    
    print(f"Target Weight (22 Asli) : {total_target_params:,} params")
    print(f"Cluster 22 Fourier Gens : {total_gen_params:,} params")
    print(f"Rasio Kompresi Baru     : {total_target_params / total_gen_params:.2f}x lebih kecil! (Sangat Ekstrem)\n")

    print("[3/4] CI/CD PIPELINE: Fast-Training 22 Fourier Generator (1000 step/layer)...")
    matrix_size = 2048
    batch_size = 8192

    for i in range(num_layers):
        optimizer = optim.Adam(generators[i].parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        for step in range(1, 1001):
            rows = torch.randint(0, matrix_size, (batch_size,))
            cols = torch.randint(0, matrix_size, (batch_size,))
            
            real = original_weights[i][rows, cols]
            pred = generators[i](rows, cols)
            
            loss = criterion(pred, real)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"✅ Fourier Layer {i:02d} Deployed | Final MSE Error: {loss.item():.6f}")

    print("\n[4/4] THE CLIFF EDGE TEST (Mencari Jurang Kematian AI)...")
    test_points = [1, 3, 5, 8, 12, 16, 22]
    
    for n in test_points:
        # Reset ke original factory weight
        for i in range(num_layers):
            model.model.layers[i].self_attn.o_proj.weight.data = original_weights[i].clone()
        
        # Brain swap progresif
        with torch.no_grad():
            for i in range(n):
                reconstructed = torch.zeros((matrix_size, matrix_size))
                for r in range(matrix_size):
                    row_idx = torch.full((matrix_size,), r, dtype=torch.long)
                    col_idx = torch.arange(matrix_size, dtype=torch.long)
                    reconstructed[r] = generators[i](row_idx, col_idx)
                model.model.layers[i].self_attn.o_proj.weight.data = reconstructed
        
        # Inference test
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        answer = tokenizer.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
        print(f"Test N={n:02d} Layers Ditukar | Output: '{answer}'")

    print("\n================ EVALUASI SELESAI ================")

if __name__ == "__main__":
    main()