"""
Eksperimen E1 — SIREN (Sinusoidal Representation Networks)

Tujuan: Mengganti arsitektur MLP standar dengan SIREN.
SIREN membuang semua aktivasi linier (GELU/ReLU) dan menggantinya dengan 
fungsi Sinus murni dengan inisialisasi bobot khusus (Sitzmann et al., 2020).
Diharapkan SIREN bisa menembus batas Spectral Bias dan menurunkan MSE 
di layer akhir (15-21) menembus angka 0.0003.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer
import math

class SineLayer(nn.Module):
    """Custom layer untuk SIREN dengan inisialisasi bobot khusus."""
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # Layer pertama diinisialisasi berbeda
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                # Layer berikutnya menggunakan rumus distribusi seragam yang disesuaikan
                self.linear.weight.uniform_(-math.sqrt(6 / self.in_features) / self.omega_0, 
                                             math.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class WijiGeneratorSiren(nn.Module):
    """
    Generator berbasis SIREN seutuhnya.
    """
    def __init__(self, matrix_size=2048, seed_dim=64, hidden_dim=256, num_layers=3):
        super().__init__()
        self.matrix_size = matrix_size
        self.seed = nn.Parameter(torch.randn(1, seed_dim))
        
        # Input: Normalized Row (1) + Normalized Col (1) + Seed
        input_dim = 2 + seed_dim
        
        layers = []
        # Layer pertama (is_first=True)
        layers.append(SineLayer(input_dim, hidden_dim, is_first=True))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim, is_first=False))
            
        # Output layer (linear murni, tidak di-sinus agar valuenya bebas)
        out_layer = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            out_layer.weight.uniform_(-math.sqrt(6 / hidden_dim) / 30, 
                                       math.sqrt(6 / hidden_dim) / 30)
        layers.append(out_layer)
        
        self.generator = nn.Sequential(*layers)

    def forward(self, row_indices, col_indices):
        # Normalisasi ke skala [-1.0, 1.0]
        r_norm = (row_indices.float() / self.matrix_size) * 2.0 - 1.0
        c_norm = (col_indices.float() / self.matrix_size) * 2.0 - 1.0
        
        r_norm = r_norm.unsqueeze(-1)
        c_norm = c_norm.unsqueeze(-1)
        
        s = self.seed.expand(r_norm.shape[0], -1)
        
        # Gabungkan koordinat 2D dengan Seed DNA
        x = torch.cat([r_norm, c_norm, s], dim=-1)
        
        return self.generator(x).squeeze(-1)


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("WIJI Eksperimen E1: Arsitektur SIREN (Target o_proj)")
    print("=" * 60)
    
    print("\n[1/4] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.float32
    )

    prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is the capital of Indonesia?\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n[2/4] Mengekstrak Matriks o_proj & Inisialisasi SIREN...")
    num_layers = 22
    original_weights = [model.model.layers[i].self_attn.o_proj.weight.data.clone() for i in range(num_layers)]
    
    generators = nn.ModuleList([
        WijiGeneratorSiren(hidden_dim=256) 
        for _ in range(num_layers)
    ])

    total_target_params = sum(t.numel() for t in original_weights)
    total_gen_params = sum(sum(p.numel() for p in g.parameters()) for g in generators)
    
    print(f"Target Weight (22 Asli) : {total_target_params:,} params")
    print(f"Cluster 22 SIREN Gens   : {total_gen_params:,} params")
    print(f"Rasio Kompresi          : {total_target_params / total_gen_params:.2f}x lebih kecil!\n")

    print("[3/4] CI/CD PIPELINE: Melatih SIREN (1000 step/layer)...")
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
            
        print(f"✅ SIREN Layer {i:02d} Deployed | Final MSE Error: {loss.item():.6f}")

    print("\n[4/4] THE CLIFF EDGE TEST (SIREN Edition)...")
    test_points = [1, 3, 5, 8, 12, 16, 22]
    
    for n in test_points:
        # Reset factory weight
        for i in range(num_layers):
            model.model.layers[i].self_attn.o_proj.weight.data = original_weights[i].clone()
        
        # Brain swap
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