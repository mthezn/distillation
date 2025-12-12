from EncoderRed import CMT_Ti
from modeling.build_sam import sam_model_registry
model = CMT_Ti(img_size=1024, output_dim=256)

# 1. Conta totale
total = sum(p.numel() for p in model.parameters())
print(f"Totale: {total:,} ({total/1e6:.2f}M)")

# 2. Trova i layer più grandi
for name, param in model.named_parameters():
    if param.numel() > 10_000_000:  # > 10M
        print(f" {name}: {param.shape} = {param.numel():,} ({param.numel()/1e6:.2f}M)")

# 3. Usa il mio script
from debug_script import debug_model_parameters

debug_model_parameters(model)