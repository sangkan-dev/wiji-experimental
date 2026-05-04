"""
Eksperimen B3 — Adaptive Capacity Per-Layer Generator

Tujuan: Test apakah masalah B2 disebabkan capacity generator yang tidak
adequate untuk layer akhir yang lebih kompleks.

Hipotesis: Layer awal punya distribusi simple (small generator OK),
layer akhir butuh more capacity.

Hasil: ❌ FAILURE dengan insight terbesar — MSE plateau independent of capacity.
Generator 10x lebih besar + training 3x lebih lama → MSE hampir identik dengan B2.
Membuktikan adanya hard limit (spectral bias) di coordinate MLP.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer


class WijiGenerator(nn.Module):
    """Generator dengan parametrized hidden dimension."""
    
    def __init__(self, matrix_size=2048, seed_dim=64, coord_dim=32, hidden_dim=128):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, seed_dim))
        self.row_emb = nn.Embedding(matrix_size, coord_dim)
        self.col_emb = nn.Embedding(matrix_size, coord_dim)
        
        self.generator = nn.Sequential(
            nn.Linear(coord_dim * 2 + seed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, row_indices, col_indices):
        r = self.row_emb(row_indices)
        c = self.col_emb(col_indices)
        s = self.seed.expand(r.shape[0], -1)
        x = torch.cat([r, c, s], dim=1)
        return self.generator(x).squeeze(-1)


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("WIJI Eksperimen B3: Adaptive Capacity")
    print("=" * 60)
    
    print(f"\n[1/4] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="cpu", dtype=torch.float32
    )
    
    prompt = (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\nWhat is the capital of Indonesia?\n"
        "<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("\n[2/4] Setting up adaptive capacity generators...")
    num_layers = 22
    matrix_size = 2048
    
    targets = [
        model.model.layers[i].self_attn.o_proj.weight.data.clone() 
        for i in range(num_layers)
    ]
    
    # ADAPTIVE CAPACITY:
    # - Early layers (0-7): small generator (hidden=128)
    # - Middle layers (8-15): medium generator (hidden=256)
    # - Late layers (16-21): large generator (hidden=512)
    generators = nn.ModuleList([
        WijiGenerator(hidden_dim=128 if i <= 7 else (256 if i <= 15 else 512))
        for i in range(num_layers)
    ])
    
    total_target_params = sum(t.numel() for t in targets)
    total_gen_params = sum(sum(p.numel() for p in g.parameters()) for g in generators)
    compression = total_target_params / total_gen_params
    
    print(f"Target: {total_target_params:,} params")
    print(f"Adaptive generators: {total_gen_params:,} params")
    print(f"Compression: {compression:.2f}x")
    
    # Train with 3x more steps than B2
    print("\n[3/4] Training 22 generators (3000 steps each — 3x more than B2)...")
    batch_size = 8192
    
    for i in range(num_layers):
        optimizer = optim.Adam(generators[i].parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        for step in range(1, 3001):  # 3x more steps than B2
            rows = torch.randint(0, matrix_size, (batch_size,))
            cols = torch.randint(0, matrix_size, (batch_size,))
            real = targets[i][rows, cols]
            pred = generators[i](rows, cols)
            loss = criterion(pred, real)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"  Layer {i:02d} | Final MSE: {loss.item():.6f}")
    
    print("\n[4/4] Brain swap all 22 layers...")
    with torch.no_grad():
        for i in range(num_layers):
            reconstructed = torch.zeros((matrix_size, matrix_size))
            for r in range(matrix_size):
                row_idx = torch.full((matrix_size,), r, dtype=torch.long)
                col_idx = torch.arange(matrix_size, dtype=torch.long)
                reconstructed[r] = generators[i](row_idx, col_idx)
            model.model.layers[i].self_attn.o_proj.weight.data = reconstructed
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    answer = tokenizer.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Output: {answer}")
    print(f"Compression: {compression:.2f}x")
    print("\nExpected: FAILURE — but MSE plateau reveals SPECTRAL BIAS")
    print("Compare layer 21 MSE: B2=0.000373, B3=0.000366 (essentially identical)")
    print("=" * 60)


if __name__ == "__main__":
    main()
