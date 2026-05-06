"""
Eksperimen G2 — WIJI-Native Tiny Transformer

Tujuan: Melatih model dari awal (Co-Training) di mana layer nn.Linear diganti
dengan WijiLinear (matriks di-generate on-the-fly dari sebuah MLP kecil).
Hipotesis: Jika loss bisa turun layaknya G1, maka WIJI berhasil!
"""

import os
import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64
block_size = 128
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cpu'
n_embd = 128
n_head = 4
n_layer = 2

torch.manual_seed(1337)

DATA_DIR = "results"
data_path = os.path.join(DATA_DIR, "input.txt")
with open(data_path, 'r', encoding='utf-8') as f: text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(100)
        for k in range(100):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- CORE WIJI ARCHITECTURE ---
class WijiLinear(nn.Module):
    """Pengganti nn.Linear. Weight-nya di-generate dari MLP, bukan disimpan!"""
    def __init__(self, in_features, out_features, gen_hidden=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Generator: Input (Row Norm, Col Norm) -> Hidden -> 1 (Value)
        self.generator = nn.Sequential(
            nn.Linear(2, gen_hidden),
            nn.GELU(),
            nn.Linear(gen_hidden, gen_hidden),
            nn.GELU(),
            nn.Linear(gen_hidden, 1)
        )
        # Pre-compute coordinates (grid) agar cepat saat training
        r_coords = (torch.arange(out_features).float() / out_features) * 2 - 1
        c_coords = (torch.arange(in_features).float() / in_features) * 2 - 1
        r_grid, c_grid = torch.meshgrid(r_coords, c_coords, indexing='ij')
        
        # Simpan grid sebagai buffer (bukan weight, tidak dilatih)
        self.register_buffer('grid', torch.stack([r_grid.flatten(), c_grid.flatten()], dim=1))
        
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # 1. Just-In-Time Generation of Weights!
        flat_weight = self.generator(self.grid).squeeze(-1)
        weight = flat_weight.view(self.out_features, self.in_features)
        
        # 2. Kalikan dengan input (MatMul)
        return F.linear(x, weight, self.bias)

# Arsitektur Transformer menggunakan WijiLinear
class HeadWiji(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = WijiLinear(n_embd, head_size)
        self.query = WijiLinear(n_embd, head_size)
        self.value = WijiLinear(n_embd, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        v = self.value(x)
        return wei @ v

class MultiHeadAttentionWiji(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([HeadWiji(head_size) for _ in range(num_heads)])
        self.proj = WijiLinear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedFowardWiji(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            WijiLinear(n_embd, 4 * n_embd),
            nn.ReLU(),
            WijiLinear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class BlockWiji(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttentionWiji(n_head, head_size)
        self.ffwd = FeedFowardWiji(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class WijiModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Embedding tetap pakai standard karena ukurannya mikroskopis (65 x 128)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[BlockWiji(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = WijiLinear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

def main():
    print("=" * 50)
    print("STARTING G2: WIJI-NATIVE CO-TRAINING")
    print("=" * 50)
    model = WijiModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Params WIJI-Native: {total_params:,}\n")

    history = []
    
    for iter in range(max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model)
            print(f"Step {iter:4d} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
            history.append(f"{iter},{losses['train']:.4f},{losses['val']:.4f}\n")

        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # <--- INI DIA! Gradient mengalir dari Loss -> Weight -> Generator DNA
        optimizer.step()

    with open(os.path.join(DATA_DIR, 'g2_loss.csv'), 'w') as f:
        f.write("step,train_loss,val_loss\n")
        f.writelines(history)

    print("\nSAMPLE GENERATION (WIJI-NATIVE):")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=200)[0].tolist()))

if __name__ == '__main__':
    main()