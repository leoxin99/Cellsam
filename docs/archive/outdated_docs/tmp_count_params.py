import sys, os
sys.path.insert(0, r'd:\AI\paper\CellSam')
sys.path.insert(0, r'd:\AI\paper\CellSam\cellSAM_source')
os.chdir(r'd:\AI\paper\CellSam')

from cellSAM.model import get_model
m = get_model()
s = m.model

enc = sum(p.numel() for p in s.image_encoder.parameters())
neck = sum(p.numel() for p in s.image_encoder.neck.parameters())
pe = sum(p.numel() for p in s.prompt_encoder.parameters())
md = sum(p.numel() for p in s.mask_decoder.parameters())
total = sum(p.numel() for p in s.parameters())

with open(r'd:\AI\paper\CellSam\docs\tmp_params.txt', 'w') as f:
    f.write(f"Encoder: {enc:,}\n")
    f.write(f"Neck: {neck:,}\n")
    f.write(f"PromptEnc: {pe:,}\n")
    f.write(f"MaskDec: {md:,}\n")
    f.write(f"Total SAM: {total:,}\n")
    f.write(f"\nStrategies:\n")
    f.write(f"Current (pe+md): {pe+md:,} ({(pe+md)/total*100:.1f}%)\n")
    f.write(f"Neck-only: {neck:,} ({neck/total*100:.2f}%)\n")
    f.write(f"Decoder-only: {md:,} ({md/total*100:.1f}%)\n")
    f.write(f"Encoder (no neck): {enc-neck:,} ({(enc-neck)/total*100:.1f}%)\n")

# Print neck layer details
f2 = open(r'd:\AI\paper\CellSam\docs\tmp_neck.txt', 'w')
f2.write("Neck layers:\n")
for name, p in s.image_encoder.neck.named_parameters():
    f2.write(f"  {name}: {list(p.shape)} = {p.numel():,}\n")
f2.close()
print("done")
