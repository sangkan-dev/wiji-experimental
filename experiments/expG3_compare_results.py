"""
Eksperimen G3 — Komparator Loss
Menganalisa dan membandingkan hasil Co-Training WIJI vs Baseline.
Pastikan pip install matplotlib / uv add matplotlib.
"""

import os
import matplotlib.pyplot as plt
import csv

def read_loss(filepath):
    steps, val_losses = [], []
    if not os.path.exists(filepath):
        return steps, val_losses
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            steps.append(int(row['step']))
            val_losses.append(float(row['val_loss']))
    return steps, val_losses

def main():
    print("Menganalisa data eksperimen...")
    DATA_DIR = "results"
    
    g1_steps, g1_loss = read_loss(os.path.join(DATA_DIR, "g1_loss.csv"))
    g2_steps, g2_loss = read_loss(os.path.join(DATA_DIR, "g2_loss.csv"))
    
    if not g1_steps or not g2_steps:
        print("Data belum lengkap. Jalankan G1 dan G2 terlebih dahulu.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(g1_steps, g1_loss, label='G1: Baseline (Standard Linear)', marker='o')
    plt.plot(g2_steps, g2_loss, label='G2: WIJI-Native (Generated Matrix)', marker='x', color='red')
    
    plt.title('Validation Loss Comparison: Baseline vs WIJI-Native')
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Loss (Lower is Better)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    out_path = os.path.join(DATA_DIR, "G3_Loss_Comparison.png")
    plt.savefig(out_path)
    print(f"\n✅ Selesai! Grafik komparasi tersimpan di: {out_path}")
    print(f"Final Val Loss G1 (Baseline) : {g1_loss[-1]:.4f}")
    print(f"Final Val Loss G2 (WIJI)     : {g2_loss[-1]:.4f}")
    
    print("\n--- KESIMPULAN MENTAH ---")
    if g2_loss[-1] <= g1_loss[-1] + 0.2:
        print("🔥 Skenario A: WIJI-Native WORKED! Loss sejajar dengan baseline.")
    elif g2_loss[-1] < g2_loss[0] - 0.5:
        print("⚠️ Skenario B: WIJI-Native converge, tapi ada degradasi quality.")
    else:
        print("❌ Skenario C: WIJI-Native gagal belajar pola bahasa secara efektif.")

if __name__ == "__main__":
    main()