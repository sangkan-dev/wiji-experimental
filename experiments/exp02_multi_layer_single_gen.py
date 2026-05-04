"""
Eksperimen 2 — Multi-Layer Single Generator

Tujuan: Test apakah 1 generator bisa handle 22 layer sekaligus
dengan tambahan layer embedding.

Hipotesis: Single generator dengan layer embedding cukup untuk
merepresentasikan semua 22 distribusi weight.

Hasil: ❌ FAILURE — Output collapse menjadi gibberish.
Insight: MSE Loss naik hanya 3x tapi output collapse total.
MSE bukan reliable predictor untuk LLM output quality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer


class WijiGeneratorMultiLayer(nn.Module):
    """Generator dengan layer embedding untuk handle multiple layers."""
    
    def __init__(self, num_layers=22, matrix_size=2048, seed_dim=64, coord_dim=32):
        super().__init__()
        self.seed = nn.Parameter(torch.randn(1, seed_dim))
        self.layer_emb = nn.Embedding(num_layers, coord_dim)
        self.row_emb = nn.Embedding(matrix_size, coord_dim)
        self.col_emb = nn.Embedding(matrix_size, coord_dim)
        
        self.generator = nn.Sequential(
            nn.Linear(coord_dim * 3 + seed_dim, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, layer_indices, row_indices, col_indices):
        l = self.layer_emb(layer_indices)
        r = self.row_emb(row_indices)
        c = self.col_emb(col_indices)
        s = self.seed.expand(r.shape[0], -1)
        x = torch.cat([l, r, c, s], dim=1)
        return self.generator(x).squeeze(-1)


def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print("=" * 60)
    print("WIJI Eksperimen 2: Multi-Layer Single Generator")
    print("=" * 60)
    
    print(f"\n[1/5] Loading {model_id}...")
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
    
    # Extract all 22 layer weights
    print("\n[2/5] Extracting 22 layer weights...")
    num_layers = 22
    matrix_size = 2048
    
    all_target_weights = torch.stack([
        model.model.layers[i].self_attn.o_proj.weight.data.clone()
        for i in range(num_layers)
    ])
    print(f"Total target params: {all_target_weights.numel():,}")
    
    # Setup generator
    generator = WijiGeneratorMultiLayer(num_layers=num_layers)
    optimizer = optim.Adam(generator.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    gen_params = sum(p.numel() for p in generator.parameters())
    compression = all_target_weights.numel() / gen_params
    print(f"Generator params: {gen_params:,}")
    print(f"Compression ratio: {compression:.2f}x (CRAZY!)")
    
    # Training
    print("\n[3/5] Training single generator for 22 layers...")
    batch_size = 8192
    num_steps = 3000
    
    for step in range(1, num_steps + 1):
        layers = torch.randint(0, num_layers, (batch_size,))
        rows = torch.randint(0, matrix_size, (batch_size,))
        cols = torch.randint(0, matrix_size, (batch_size,))
        
        real_values = all_target_weights[layers, rows, cols]
        predicted_values = generator(layers, rows, cols)
        
        loss = criterion(predicted_values, real_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"  Step {step:4d}/{num_steps} | MSE: {loss.item():.6f}")
    
    # Reconstruct all 22 layers
    print("\n[4/5] Reconstructing all 22 layers...")
    with torch.no_grad():
        for l in range(num_layers):
            reconstructed = torch.zeros((matrix_size, matrix_size))
            layer_idx = torch.full((matrix_size,), l, dtype=torch.long)
            
            for r in range(matrix_size):
                row_idx = torch.full((matrix_size,), r, dtype=torch.long)
                col_idx = torch.arange(matrix_size, dtype=torch.long)
                reconstructed[r] = generator(layer_idx, row_idx, col_idx)
            
            model.model.layers[l].self_attn.o_proj.weight.data = reconstructed
    
    # Test
    print("\n[5/5] Testing reconstructed model...")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    answer = tokenizer.decode(out[0], skip_special_tokens=True).split("<|assistant|>")[-1].strip()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Output: {answer}")
    print(f"Compression: {compression:.2f}x")
    print(f"Final MSE: {loss.item():.6f}")
    print("\nExpected: FAILURE — single generator insufficient for 22 layers")
    print("=" * 60)


if __name__ == "__main__":
    main()
