"""
Eksperimen F1 — Mixture of Generators (MoG / Sharding)

Tujuan: Memecah matriks 2048x2048 menjadi 16 blok kecil (ukuran 512x512).
Alih-alih 1 fungsi menghafal seluruh matriks, kita memiliki 16 "Expert DNA" 
yang masing-masing hanya fokus menghafal area kecilnya saja.
Diharapkan ini memecahkan masalah kapasitas dan lokalisasi spasial.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer

class WijiGeneratorMoG(nn.Module):
    """
    Generator dengan internal routing (Database Sharding style).
    Memecah matriks menjadi grid (contoh: 4x4 = 16 chunk).
    """
    def __init__(self, matrix_size=2048, chunks_per_dim=4, 
                 expert_dim=64, coord_dim=32, hidden_dim=256):
        super().__init__()
        self.matrix_size = matrix_size
        self.chunks_per_dim = chunks_per_dim
        self.chunk_size = matrix_size // chunks_per_dim  # 2048 // 4 = 512
        
        num_experts = chunks_per_dim * chunks_per_dim  # 16 Experts
        
        # 1. Tabel DNA untuk para "Spesialis"
        self.expert_embeddings = nn.Embedding(num_experts, expert_dim)
        
        # 2. Tabel Kordinat Lokal (hanya perlu menghafal 0 - 511, bukan 0 - 2047)
        self.local_row_emb = nn.Embedding(self.chunk_size, coord_dim)
        self.local_col_emb = nn.Embedding(self.chunk_size, coord_dim)
        
        # 3. Otak Eksekutor
        input_dim = expert_dim + (coord_dim * 2)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, row_indices, col_indices):
        # A. ROUTING LOGIC (Mencari tahu koordinat ini jatahnya Expert yang mana)
        row_chunk = row_indices // self.chunk_size
        col_chunk = col_indices // self.chunk_size
        chunk_id = (row_chunk * self.chunks_per_dim) + col_chunk
        
        # B. LOKALISASI (Mengubah koordinat global ke kordinat lokal 0-511)
        local_row = row_indices % self.chunk_size
        local_col = col_indices % self.chunk_size
        
        # C. FETCH DATA
        expert_dna = self.expert_embeddings(chunk_id)
        r_emb = self.local_row_emb(local_row)
        c_emb = self.local_col_emb(local_col)
        
        # D. EXECUTE
        x = torch.cat([expert_dna, r_emb, c_emb], dim=-1)
        return self.mlp(x).squeeze(-1)


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("WIJI Eksperimen F1: Mixture of Generators (MoG Sharding)")
    print("=" * 60)
    
    print("\n[1/4] Loading Model & Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.float32
    )

    prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nWhat is the capital of Indonesia?\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n[2/4] Mengekstrak Matriks o_proj & Inisialisasi MoG Router...")
    num_layers = 22
    original_weights = [model.model.layers[i].self_attn.o_proj.weight.data.clone() for i in range(num_layers)]
    
    generators = nn.ModuleList([
        WijiGeneratorMoG(matrix_size=2048, chunks_per_dim=4) 
        for _ in range(num_layers)
    ])

    total_target_params = sum(t.numel() for t in original_weights)
    total_gen_params = sum(sum(p.numel() for p in g.parameters()) for g in generators)
    
    print(f"Target Weight (22 Asli) : {total_target_params:,} params")
    print(f"Cluster 22 MoG Gens     : {total_gen_params:,} params")
    print(f"Rasio Kompresi          : {total_target_params / total_gen_params:.2f}x lebih kecil!\n")

    print("[3/4] CI/CD PIPELINE: Melatih MoG Cluster (1000 step/layer)...")
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
            
        print(f"✅ MoG Cluster Layer {i:02d} Deployed | Final MSE Error: {loss.item():.6f}")

    print("\n[4/4] THE CLIFF EDGE TEST (MoG Edition)...")
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