"""
Eksperimen D1 — FFN Layer Target (Ganti Sasaran)

Tujuan: Menguji hipotesis apakah distribusi matriks FFN (down_proj) 
lebih mudah dipelajari oleh Fourier Generator dibandingkan matriks Attention (o_proj).
FFN sering dianggap sebagai "Key-Value Memory" yang mungkin memiliki 
spatial locality yang lebih ramah terhadap MLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

class WijiGeneratorFourierDynamic(nn.Module):
    """
    Generator Fourier yang dioptimalkan untuk matriks persegi panjang.
    """
    def __init__(self, max_rows, max_cols, seed_dim=64, num_freq=16, hidden_dim=256):
        super().__init__()
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.num_freq = num_freq
        self.seed = nn.Parameter(torch.randn(1, seed_dim))
        
        input_dim = (num_freq * 2 * 2) + seed_dim
        
        self.generator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, row_indices, col_indices):
        r_norm = (row_indices.float() / self.max_rows) * 2.0 - 1.0
        c_norm = (col_indices.float() / self.max_cols) * 2.0 - 1.0
        
        r_norm = r_norm.unsqueeze(-1)
        c_norm = c_norm.unsqueeze(-1)

        freqs = (2 ** torch.arange(self.num_freq, device=r_norm.device)).float()
        
        r_freq = r_norm * freqs * math.pi
        c_freq = c_norm * freqs * math.pi
        
        r_fourier = torch.cat([r_freq.sin(), r_freq.cos()], dim=-1)
        c_fourier = torch.cat([c_freq.sin(), c_freq.cos()], dim=-1)
        
        s = self.seed.expand(r_norm.shape[0], -1)
        x = torch.cat([r_fourier, c_fourier, s], dim=-1)
        
        return self.generator(x).squeeze(-1)


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("WIJI Eksperimen D1: Menyerang FFN Layer (down_proj)")
    print("=" * 60)
    
    print("\n[1/4] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.float32
    )

    prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is the capital of Indonesia?\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n[2/4] Mengekstrak Matriks FFN (down_proj)...")
    num_layers = 22
    
    # KITA GANTI TARGETNYA DI SINI:
    original_weights = [model.model.layers[i].mlp.down_proj.weight.data.clone() for i in range(num_layers)]
    
    # Ambil ukuran dinamis (2048 x 5632)
    max_rows, max_cols = original_weights[0].shape
    print(f"Bentuk Matriks FFN: {max_rows} baris x {max_cols} kolom")

    generators = nn.ModuleList([
        WijiGeneratorFourierDynamic(max_rows=max_rows, max_cols=max_cols) 
        for _ in range(num_layers)
    ])

    total_target_params = sum(t.numel() for t in original_weights)
    total_gen_params = sum(sum(p.numel() for p in g.parameters()) for g in generators)
    
    print(f"Target Weight (22 Asli) : {total_target_params:,} params")
    print(f"Cluster 22 Fourier Gens : {total_gen_params:,} params")
    print(f"Rasio Kompresi          : {total_target_params / total_gen_params:.2f}x lebih kecil!\n")

    print("[3/4] CI/CD PIPELINE: Melatih Fourier di Area FFN (1000 step/layer)...")
    batch_size = 8192

    for i in range(num_layers):
        optimizer = optim.Adam(generators[i].parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        for step in range(1, 1001):
            rows = torch.randint(0, max_rows, (batch_size,))
            cols = torch.randint(0, max_cols, (batch_size,))
            
            real = original_weights[i][rows, cols]
            pred = generators[i](rows, cols)
            
            loss = criterion(pred, real)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        print(f"✅ FFN Fourier Layer {i:02d} Deployed | Final MSE: {loss.item():.6f}")

    print("\n[4/4] THE CLIFF EDGE TEST (Di Area FFN)...")
    test_points = [1, 3, 5, 8, 12, 16, 22]
    
    for n in test_points:
        # Reset ke original factory weight
        for i in range(num_layers):
            model.model.layers[i].mlp.down_proj.weight.data = original_weights[i].clone()
        
        # Brain swap progresif pada FFN
        with torch.no_grad():
            for i in range(n):
                reconstructed = torch.zeros((max_rows, max_cols))
                # Harus proses per batch kolom agar RAM tidak meledak (karena 5632 kolom itu besar)
                for r in range(max_rows):
                    row_idx = torch.full((max_cols,), r, dtype=torch.long)
                    col_idx = torch.arange(max_cols, dtype=torch.long)
                    reconstructed[r] = generators[i](row_idx, col_idx)
                model.model.layers[i].mlp.down_proj.weight.data = reconstructed
        
        # Inference test
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        answer = tokenizer.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
        print(f"Test N={n:02d} Layers Ditukar | Output: '{answer}'")

    print("\n================ EVALUASI SELESAI ================")

if __name__ == "__main__":
    main()