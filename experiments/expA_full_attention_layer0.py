"""
Eksperimen Diagnostik A — Full Attention Layer 0

Tujuan: Test apakah masalah di Eksperimen 2 (multi-layer collapse)
disebabkan oleh single generator yang tidak bisa handle multiple
components, atau memang fenomena multi-layer compounding.

Hipotesis: Kalau 1 generator handle Q/K/V/O di 1 layer berhasil,
masalah di Eksperimen 2 adalah multi-layer, bukan multi-component.

Hasil: ✅ SUCCESS — Output preserved meski MSE 6x lebih tinggi dari Exp 1.
Konfirmasi: masalah Exp 2 adalah multi-layer compounding error.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer


class WijiGeneratorMultiComponent(nn.Module):
    """Generator dengan component embedding untuk Q/K/V/O."""
    
    def __init__(self, num_components=4, max_rows=2048, max_cols=2048, 
                 seed_dim=64, coord_dim=32):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, seed_dim))
        self.comp_emb = nn.Embedding(num_components, coord_dim)
        self.row_emb = nn.Embedding(max_rows, coord_dim)
        self.col_emb = nn.Embedding(max_cols, coord_dim)
        
        self.generator = nn.Sequential(
            nn.Linear(coord_dim * 3 + seed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, comp_indices, row_indices, col_indices):
        comp = self.comp_emb(comp_indices)
        r = self.row_emb(row_indices)
        c = self.col_emb(col_indices)
        s = self.seed.expand(r.shape[0], -1)
        x = torch.cat([comp, r, c, s], dim=1)
        return self.generator(x).squeeze(-1)


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("WIJI Eksperimen Diagnostik A: Full Attention Layer 0")
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
    
    # Extract all 4 attention matrices
    print("\n[2/4] Extracting Q/K/V/O matrices from layer 0...")
    attn = model.model.layers[0].self_attn
    targets = [
        attn.q_proj.weight.data.clone(),  # ID 0
        attn.k_proj.weight.data.clone(),  # ID 1
        attn.v_proj.weight.data.clone(),  # ID 2
        attn.o_proj.weight.data.clone()   # ID 3
    ]
    total_target_params = sum(t.numel() for t in targets)
    
    generator = WijiGeneratorMultiComponent()
    optimizer = optim.Adam(generator.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    gen_params = sum(p.numel() for p in generator.parameters())
    compression = total_target_params / gen_params
    print(f"Total target params: {total_target_params:,}")
    print(f"Generator params: {gen_params:,}")
    print(f"Compression ratio: {compression:.2f}x")
    
    # Training
    print("\n[3/4] Training...")
    batch_size = 8192
    num_steps = 3000
    
    for step in range(1, num_steps + 1):
        comps = torch.randint(0, 4, (batch_size,))
        rows = torch.zeros(batch_size, dtype=torch.long)
        cols = torch.zeros(batch_size, dtype=torch.long)
        real_values = torch.zeros(batch_size, dtype=torch.float32)
        
        # Map coordinates per component (matrices may have different sizes)
        for i in range(4):
            mask = (comps == i)
            num_c = mask.sum().item()
            if num_c > 0:
                max_r, max_c = targets[i].shape
                r = torch.randint(0, max_r, (num_c,))
                c = torch.randint(0, max_c, (num_c,))
                rows[mask] = r
                cols[mask] = c
                real_values[mask] = targets[i][r, c]
        
        predicted_values = generator(comps, rows, cols)
        loss = criterion(predicted_values, real_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"  Step {step:4d}/{num_steps} | MSE: {loss.item():.6f}")
    
    # Brain swap all 4 attention matrices in layer 0
    print("\n[4/4] Brain swap layer 0 (all 4 attention matrices)...")
    with torch.no_grad():
        projs = [attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj]
        for i, proj in enumerate(projs):
            max_r, max_c = targets[i].shape
            reconstructed = torch.zeros((max_r, max_c))
            comp_idx = torch.full((max_c,), i, dtype=torch.long)
            
            for r in range(max_r):
                row_idx = torch.full((max_c,), r, dtype=torch.long)
                col_idx = torch.arange(max_c, dtype=torch.long)
                reconstructed[r] = generator(comp_idx, row_idx, col_idx)
            
            proj.weight.data = reconstructed
    
    # Test
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    answer = tokenizer.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Output: {answer}")
    print(f"Compression: {compression:.2f}x")
    print(f"Final MSE: {loss.item():.6f}")
    print("\nExpected: SUCCESS — confirms Exp 2 failure was multi-layer issue")
    print("=" * 60)


if __name__ == "__main__":
    main()
