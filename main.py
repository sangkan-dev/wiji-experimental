import torch
from transformers import AutoModelForCausalLM

def main():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"Memuat {model_id} ke RAM... (Tunggu sebentar)")
    
    # Kita load modelnya ke CPU saja karena kita cuma mau ekstrak weight-nya
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="cpu", 
        torch_dtype=torch.float32
    )

    # ---------------------------------------------------------
    # TARGET KORBAN: Matriks 'o_proj' (Output Projection) dari Layer 0
    # Ini adalah matriks yang akan coba kita generate dari "seed" nantinya
    # ---------------------------------------------------------
    target_weight = model.model.layers[0].self_attn.o_proj.weight.data

    print("\n=== EKSTRAKSI BERHASIL ===")
    print(f"Target Matriks  : Layer 0 -> o_proj")
    print(f"Bentuk (Shape)  : {target_weight.shape} (Baris x Kolom)")
    print(f"Total Angka     : {target_weight.numel():,} parameter")
    print(f"Contoh 5 Angka  : {target_weight.flatten()[:5].tolist()}")
    print("==========================\n")
    
    # Nanti di step selanjutnya, target_weight inilah yang akan kita 
    # jadikan "kunci jawaban" (ground truth) saat melatih model generator fraktal kita.

if __name__ == "__main__":
    main()