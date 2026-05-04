"""
Eksperimen B4 — Cliff Edge Test

Tujuan: Mencari titik dimana model collapse — berapa banyak layer 
yang bisa di-swap sebelum semantic coherence hilang.

Hipotesis: Akan ada gradual degradation seiring jumlah swap.

Hasil: ❌ Cliff edge sangat tajam ditemukan antara N=1 dan N=3.
Insight kritis:
- N=1: Perfect output
- N=3: Partial collapse  
- N=5-16: Empty output (probability collapse)
- N=22: Gibberish (internal consistency restored)

Pattern N=22 lebih baik dari N=5-16 sangat informatif:
internal consistency matters more than absolute correctness.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer


class WijiGenerator(nn.Module):
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
    print("WIJI Eksperimen B4: Cliff Edge Test")
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
    
    # Backup original weights
    print("\n[2/4] Backing up original weights and initializing generators...")
    num_layers = 22
    matrix_size = 2048
    
    original_weights = [
        model.model.layers[i].self_attn.o_proj.weight.data.clone() 
        for i in range(num_layers)
    ]
    
    # Adaptive capacity: layer akhir dapet generator lebih besar
    generators = nn.ModuleList([
        WijiGenerator(hidden_dim=128 if i <= 7 else (256 if i <= 15 else 512))
        for i in range(num_layers)
    ])
    
    # Train all generators (fast pass — 1000 steps each)
    print("\n[3/4] Fast-training 22 generators (1000 steps each)...")
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
        
        print(f"  Layer {i:02d} trained | Final MSE: {loss.item():.6f}")
    
    # Cliff edge test
    print("\n[4/4] CLIFF EDGE TEST — progressive layer swap...")
    test_points = [1, 3, 5, 8, 12, 16, 22]
    results = []
    
    for n in test_points:
        # Restore all original weights
        for i in range(num_layers):
            model.model.layers[i].self_attn.o_proj.weight.data = original_weights[i].clone()
        
        # Swap layer 0 to n-1 with generated weights
        with torch.no_grad():
            for i in range(n):
                reconstructed = torch.zeros((matrix_size, matrix_size))
                for r in range(matrix_size):
                    row_idx = torch.full((matrix_size,), r, dtype=torch.long)
                    col_idx = torch.arange(matrix_size, dtype=torch.long)
                    reconstructed[r] = generators[i](row_idx, col_idx)
                model.model.layers[i].self_attn.o_proj.weight.data = reconstructed
        
        # Test inference
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        answer = tokenizer.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
        results.append((n, answer))
        print(f"  N={n:02d} layers swapped | Output: '{answer}'")
    
    # Summary
    print("\n" + "=" * 60)
    print("CLIFF EDGE RESULTS")
    print("=" * 60)
    for n, answer in results:
        status = "✅" if "Jakarta" in answer else "❌"
        print(f"{status} N={n:02d}: {answer[:60]}")
    print("=" * 60)
    print("\nExpected pattern: Cliff edge between N=1 and N=3")
    print("Counterintuitive: N=22 may produce output while N=5-16 are empty")


if __name__ == "__main__":
    main()
