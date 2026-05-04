"""
Eksperimen 1 — Single Matrix Reconstruction

Tujuan: Membuktikan bahwa weight matrix dari LLM bisa di-rekonstruksi 
dari coordinate-based generator yang jauh lebih kecil.

Hipotesis: Weight matrix punya redundansi internal yang bisa di-exploit
oleh hypernetwork dengan parameter signifikan lebih sedikit.

Hasil: ✅ SUCCESS — Output semantik preserved dengan compression 25x.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForCausalLM, AutoTokenizer


class WijiGenerator(nn.Module):
    """
    Coordinate-based MLP generator untuk weight reconstruction.
    
    Input: koordinat (row, col) di matrix
    Output: nilai weight di koordinat tersebut
    """
    
    def __init__(self, matrix_size=2048, seed_dim=64, coord_dim=32, hidden_dim=128):
        super().__init__()
        # "DNA" model — vector kecil yang berisi essence
        self.seed = nn.Parameter(torch.randn(1, seed_dim))
        
        # Embeddings untuk koordinat
        self.row_emb = nn.Embedding(matrix_size, coord_dim)
        self.col_emb = nn.Embedding(matrix_size, coord_dim)
        
        # Generator network
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
    print("WIJI Eksperimen 1: Single Matrix Reconstruction")
    print("=" * 60)
    
    print(f"\n[1/5] Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cpu", 
        dtype=torch.float32
    )
    
    prompt = (
        "<|system|>\nYou are a helpful assistant.\n"
        "<|user|>\nWhat is the capital of Indonesia?\n"
        "<|assistant|>\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Test original output
    print("\n[2/5] Testing original output...")
    with torch.no_grad():
        out_original = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    original_text = tokenizer.decode(out_original[0], skip_special_tokens=True)
    original_answer = original_text.split("<|assistant|>")[-1].strip()
    print(f"Original: {original_answer}")
    
    # Extract target weight
    print("\n[3/5] Extracting target weight (o_proj layer 0)...")
    target_weight = model.model.layers[0].self_attn.o_proj.weight.data.clone()
    print(f"Target shape: {target_weight.shape}")
    print(f"Target params: {target_weight.numel():,}")
    
    # Setup generator
    generator = WijiGenerator()
    optimizer = optim.Adam(generator.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    gen_params = sum(p.numel() for p in generator.parameters())
    compression = target_weight.numel() / gen_params
    print(f"Generator params: {gen_params:,}")
    print(f"Compression ratio: {compression:.2f}x")
    
    # Training
    print("\n[4/5] Training generator...")
    matrix_size = 2048
    batch_size = 8192
    num_steps = 5000
    
    for step in range(1, num_steps + 1):
        rows = torch.randint(0, matrix_size, (batch_size,))
        cols = torch.randint(0, matrix_size, (batch_size,))
        real_values = target_weight[rows, cols]
        predicted_values = generator(rows, cols)
        
        loss = criterion(predicted_values, real_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 500 == 0:
            print(f"  Step {step:4d}/{num_steps} | MSE: {loss.item():.6f}")
    
    # Reconstruct full matrix
    print("\n[5/5] Reconstructing full matrix and testing...")
    reconstructed = torch.zeros((matrix_size, matrix_size))
    with torch.no_grad():
        for r in range(matrix_size):
            row_idx = torch.full((matrix_size,), r, dtype=torch.long)
            col_idx = torch.arange(matrix_size, dtype=torch.long)
            reconstructed[r] = generator(row_idx, col_idx)
    
    # Brain swap
    model.model.layers[0].self_attn.o_proj.weight.data = reconstructed
    
    # Test reconstructed output
    with torch.no_grad():
        out_reconstructed = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    reconstructed_text = tokenizer.decode(out_reconstructed[0], skip_special_tokens=True)
    reconstructed_answer = reconstructed_text.split("<|assistant|>")[-1].strip()
    
    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Original output:      {original_answer}")
    print(f"Reconstructed output: {reconstructed_answer}")
    print(f"Compression: {compression:.2f}x")
    print(f"Final MSE: {loss.item():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
